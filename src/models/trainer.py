"""
Training loop manager with validation, early stopping, LR scheduling, and metric logging.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.logger import setup_logger

logger = setup_logger("trainer")


class ModelTrainer:
    """Encapsulates PyTorch training loop, validation monitoring, early stopping, and model checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        device: Union[str, torch.device] = "auto",
    ) -> None:
        """Initializes ModelTrainer.

        Args:
            model: PyTorch model instance.
            loss_fn: Custom or standard loss function.
            optimizer: PyTorch optimizer instance.
            lr_scheduler: Optional learning rate scheduler.
            device: Training device ('cpu', 'cuda', 'mps', or 'auto').
        """
        self.device = self._resolve_device(device)
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.train_history: List[float] = []
        self.val_history: List[float] = []

    @staticmethod
    def _resolve_device(device: Union[str, torch.device]) -> torch.device:
        """Resolves target training device, automatically selecting best hardware if 'auto'.

        Args:
            device: Requested device string or torch.device instance.

        Returns:
            torch.device: Resolved target device instance.
        """
        if isinstance(device, str) and device.lower() == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        patience: int = 10,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, List[float]]:
        """Executes full training loop over specified epochs.

        Args:
            train_loader: DataLoader for training dataset.
            val_loader: Optional DataLoader for validation dataset.
            epochs: Maximum number of epochs to train.
            patience: Early stopping patience.
            checkpoint_path: Path to save best model weights.

        Returns:
            Dict[str, List[float]]: History dictionary containing 'train_loss' and 'val_loss'.
        """
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # Training Step
            self.model.train()
            train_loss = 0.0
            total_train_samples = 0

            for batch in train_loader:
                if len(batch) == 2:
                    x_b, y_b = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    x_b = batch[0].to(self.device)
                    y_b = None

                self.optimizer.zero_grad()

                # Model forward pass handling different output formats
                out = self.model(x_b)
                if isinstance(out, tuple):
                    # MDN tuple output (pi, mu, sigma)
                    pi, mu, sigma = out
                    loss = self.loss_fn(pi, mu, sigma, y_b)
                else:
                    loss = self.loss_fn(out, y_b)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                batch_size = x_b.size(0)
                train_loss += loss.item() * batch_size
                total_train_samples += batch_size

            avg_train_loss = train_loss / max(1, total_train_samples)
            self.train_history.append(avg_train_loss)

            # Validation Step
            avg_val_loss = avg_train_loss
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                total_val_samples = 0

                with torch.no_grad():
                    for batch in val_loader:
                        x_b, y_b = batch[0].to(self.device), batch[1].to(self.device)
                        out = self.model(x_b)

                        if isinstance(out, tuple):
                            pi, mu, sigma = out
                            v_loss = self.loss_fn(pi, mu, sigma, y_b)
                        else:
                            v_loss = self.loss_fn(out, y_b)

                        batch_size = x_b.size(0)
                        val_loss += v_loss.item() * batch_size
                        total_val_samples += batch_size

                avg_val_loss = val_loss / max(1, total_val_samples)
                self.val_history.append(avg_val_loss)

                if self.lr_scheduler is not None:
                    if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.lr_scheduler.step(avg_val_loss)
                    else:
                        self.lr_scheduler.step()

                # Early Stopping & Checkpoint Check
                if avg_val_loss < best_val_loss - 1e-4:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    if checkpoint_path:
                        path = Path(checkpoint_path)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(self.model.state_dict(), path)
                else:
                    patience_counter += 1

                if epoch % 5 == 0 or epoch == epochs:
                    logger.info(
                        f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
                    )

                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
                    break

        return {"train_loss": self.train_history, "val_loss": self.val_history}
