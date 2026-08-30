"""
Command-Line Interface (CLI) Pipeline Runner for ev-flex-ml.

Supports training probabilistic deep learning models, executing counterfactual backtests, and serving REST APIs.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.data_loader import SyntheticDataGenerator
from src.data.preprocessor import SessionPreprocessor
from src.evaluation.backtest import BacktestEngine
from src.models.mdn_network import MixtureDensityNetwork, mdn_loss_function
from src.models.quantile_tcn import QuantileTCN, pinball_loss
from src.models.trainer import ModelTrainer
from src.utils.config import load_config
from src.utils.logger import set_seed, setup_logger

logger = setup_logger("run_pipeline")


def run_backtest(args: argparse.Namespace) -> None:
    """Executes counterfactual backtest benchmark across Unmanaged, TOU, and Smart MPC strategies.

    Args:
        args: Parsed CLI command arguments.
    """
    set_seed(args.seed)

    config_path = args.config if args.config else "configs/simulation_config.yaml"
    config = load_config(config_path) if Path(config_path).exists() else {}

    feeder_cap = config.get("feeder", {}).get("max_capacity_kw", 150.0)
    dt_hours = config.get("time", {}).get("step_minutes", 15) / 60.0

    logger.info(f"Generating synthetic fleet session logs ({args.sessions} sessions) & EPEX SPOT prices...")
    gen = SyntheticDataGenerator(seed=args.seed)
    raw_sessions = gen.generate_sessions(num_sessions=args.sessions)
    df_prices = gen.generate_price_signal(num_days=max(3, int(args.steps * dt_hours / 24.0) + 1), step_minutes=int(dt_hours * 60))

    engine = BacktestEngine(feeder_capacity_kw=feeder_cap, dt_hours=dt_hours)
    df_summary, raw_results = engine.run_backtest_comparison(
        sessions=raw_sessions,
        price_signal=df_prices["price_eur_kwh"].values,
        total_steps=args.steps,
    )

    print("\n" + "=" * 80)
    print(" COUNTERFACTUAL BACKTEST BENCHMARK SUMMARY TABLE ")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80 + "\n")


def run_train(args: argparse.Namespace) -> None:
    """Trains probabilistic deep learning model (MDN or Quantile TCN) and saves checkpoints.

    Args:
        args: Parsed CLI command arguments.
    """
    set_seed(args.seed)

    config_path = args.config if args.config else "configs/model_config.yaml"
    config = load_config(config_path) if Path(config_path).exists() else {}

    gen = SyntheticDataGenerator(seed=args.seed)
    df = gen.generate_sessions(num_sessions=300)

    preprocessor = SessionPreprocessor()
    X_scaled, y_scaled = preprocessor.fit_transform(df)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    if args.model.lower() == "mdn":
        model_cfg = config.get("mdn", {})
        model = MixtureDensityNetwork(
            input_dim=X_scaled.shape[1],
            hidden_dims=model_cfg.get("hidden_dims", [64, 128, 64]),
            num_mixtures=model_cfg.get("num_mixtures", 5),
            output_dim=2,
            dropout=model_cfg.get("dropout", 0.1),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg.get("learning_rate", 1e-3))
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=model_cfg.get("lr_scheduler", {}).get("factor", 0.5),
            patience=model_cfg.get("lr_scheduler", {}).get("patience", 5),
        )
        loss_fn = mdn_loss_function
        save_path = checkpoint_dir / "mdn_checkpoint.pt"

        train_loader = DataLoader(train_ds, batch_size=model_cfg.get("batch_size", 32), shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=model_cfg.get("batch_size", 32))

    else:
        model_cfg = config.get("quantile_tcn", {})
        seq_len = model_cfg.get("sequence_length", 24)
        model = QuantileTCN(
            input_dim=X_scaled.shape[1],
            num_channels=model_cfg.get("num_channels", [32, 64, 64, 32]),
            kernel_size=model_cfg.get("kernel_size", 3),
            quantiles=model_cfg.get("quantiles", [0.1, 0.5, 0.9]),
        )
        # Reshape for 1D convolution: [batch_size, input_dim, seq_len]
        X_tcn = X_tensor.unsqueeze(-1).repeat(1, 1, seq_len)
        y_tcn = y_tensor[:, 0:1].unsqueeze(-1).repeat(1, 1, seq_len)

        tcn_dataset = TensorDataset(X_tcn, y_tcn)
        tcn_train_size = int(0.8 * len(tcn_dataset))
        tcn_val_size = len(tcn_dataset) - tcn_train_size
        tcn_train_ds, tcn_val_ds = torch.utils.data.random_split(tcn_dataset, [tcn_train_size, tcn_val_size])

        train_loader = DataLoader(tcn_train_ds, batch_size=model_cfg.get("batch_size", 32), shuffle=True)
        val_loader = DataLoader(tcn_val_ds, batch_size=model_cfg.get("batch_size", 32))

        optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg.get("learning_rate", 1e-3))
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=model_cfg.get("lr_scheduler", {}).get("factor", 0.5),
            patience=model_cfg.get("lr_scheduler", {}).get("patience", 5),
        )
        loss_fn = pinball_loss
        save_path = checkpoint_dir / "tcn_checkpoint.pt"

    trainer = ModelTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device="auto",
    )
    logger.info(f"Starting training for {args.model.upper()} model over {args.epochs} epochs...")

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        checkpoint_path=save_path,
    )

    logger.info(f"Model saved successfully to {save_path}")


def run_serve(args: argparse.Namespace) -> None:
    """Launches FastAPI Uvicorn REST API server.

    Args:
        args: Parsed CLI command arguments.
    """
    import uvicorn
    logger.info(f"Starting FastAPI web server on http://{args.host}:{args.port}")
    uvicorn.run("api.main:app", host=args.host, port=args.port, reload=True)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ev-flex-ml: Smart EV Fleet Demand Flexibility & Charging Session Optimization CLI Runner"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["backtest", "train", "serve"],
        default="backtest",
        help="Pipeline execution mode: 'backtest', 'train', or 'serve'.",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to custom YAML config file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    # Backtest arguments
    parser.add_argument("--sessions", type=int, default=60, help="Number of EV sessions for backtesting.")
    parser.add_argument("--steps", type=int, default=192, help="Total 15-min time steps for backtesting.")

    # Train arguments
    parser.add_argument("--model", type=str, choices=["mdn", "tcn"], default="mdn", help="Model type to train.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--checkpoint-dir", type=str, default="models/checkpoints", help="Directory to save checkpoints.")

    # Serve arguments
    parser.add_argument("--host", type=str, default="0.0.0.0", help="FastAPI host address.")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI port.")

    args = parser.parse_args()

    if args.mode == "backtest":
        run_backtest(args)
    elif args.mode == "train":
        run_train(args)
    elif args.mode == "serve":
        run_serve(args)


if __name__ == "__main__":
    main()
