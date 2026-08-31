"""
FastAPI application for EV Fleet Demand Flexibility & Charging Dispatch Optimization.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    DispatchRequest,
    DispatchResponse,
    DispatchScheduleItem,
    HealthCheck,
    MDNPredictionRequest,
    MDNPredictionResponse,
    QuantilePredictionResponse,
)
from src.data.preprocessor import SessionPreprocessor
from src.models.mdn_network import MixtureDensityNetwork
from src.models.quantile_tcn import QuantileTCN
from src.optimization.milp_scheduler import MILPScheduler
from src.optimization.mpc_controller import MPCController
from src.utils.logger import setup_logger

logger = setup_logger("api_main")

MDN_CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "models" / "checkpoints" / "mdn_checkpoint.pt"
_mdn_model: Optional[MixtureDensityNetwork] = None


def get_mdn_model() -> Optional[MixtureDensityNetwork]:
    """Loads and caches the MDN PyTorch checkpoint if available."""
    global _mdn_model
    if _mdn_model is None and MDN_CHECKPOINT_PATH.exists():
        try:
            model = MixtureDensityNetwork(
                input_dim=12,
                hidden_dims=[64, 128, 64],
                num_mixtures=5,
                output_dim=2,
            )
            state_dict = torch.load(MDN_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            _mdn_model = model
            logger.info("Successfully loaded MDN checkpoint from %s", MDN_CHECKPOINT_PATH)
        except Exception as e:
            logger.warning("Failed to load MDN checkpoint: %s. Using heuristic fallback.", e)
            _mdn_model = None
    return _mdn_model


app = FastAPI(
    title="ev-flex-ml API",
    description="Smart EV Fleet Demand Flexibility & Charging Session Optimization REST API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthCheck, tags=["System"])
def health_check() -> HealthCheck:
    """Returns application health status and current timestamp."""
    return HealthCheck(
        status="ok",
        version="0.1.0",
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict/mdn", response_model=List[MDNPredictionResponse], tags=["ML Forecasting"])
def predict_mdn(request: MDNPredictionRequest) -> List[MDNPredictionResponse]:
    """Predicts probabilistic distributions (mean and std) for session duration and energy requirement using MDN."""
    if not request.sessions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session list cannot be empty.",
        )

    model = get_mdn_model()
    responses: List[MDNPredictionResponse] = []

    if model is not None:
        try:
            session_dicts = [sess.model_dump() for sess in request.sessions]
            df = pd.DataFrame(session_dicts)
            preprocessor = SessionPreprocessor()
            df_features = preprocessor.extract_features(df)
            X = df_features[preprocessor.feature_cols].values.astype(np.float32)
            x_tensor = torch.tensor(X, dtype=torch.float32)

            with torch.no_grad():
                mean_pred, std_pred = model.predict_distribution(x_tensor)
                mean_arr = mean_pred.cpu().numpy()
                std_arr = std_pred.cpu().numpy()

            for i, sess in enumerate(request.sessions):
                dur_mean = max(0.5, float(mean_arr[i, 0]))
                dur_std = max(0.1, float(std_arr[i, 0]))
                energy_mean = max(0.5, float(mean_arr[i, 1]))
                energy_std = max(0.1, float(std_arr[i, 1]))

                responses.append(
                    MDNPredictionResponse(
                        session_id=sess.session_id,
                        expected_duration_hours=round(dur_mean, 2),
                        std_duration_hours=round(dur_std, 2),
                        expected_energy_kwh=round(energy_mean, 2),
                        std_energy_kwh=round(energy_std, 2),
                    )
                )
            return responses
        except Exception as e:
            logger.warning("MDN inference failed: %s. Falling back to heuristics.", e)
            responses.clear()

    # Heuristic fallback
    for sess in request.sessions:
        arr_dt = pd.to_datetime(sess.arrival_time)
        dep_dt = pd.to_datetime(sess.departure_time)
        dur_hrs = max(1.0, (dep_dt - arr_dt).total_seconds() / 3600.0)

        responses.append(
            MDNPredictionResponse(
                session_id=sess.session_id,
                expected_duration_hours=round(dur_hrs, 2),
                std_duration_hours=round(0.15 * dur_hrs, 2),
                expected_energy_kwh=round(sess.required_energy_kwh, 2),
                std_energy_kwh=round(0.10 * sess.required_energy_kwh, 2),
            )
        )

    return responses


@app.post("/predict/quantile", response_model=QuantilePredictionResponse, tags=["ML Forecasting"])
def predict_quantile(history_hours: int = 24) -> QuantilePredictionResponse:
    """Generates multi-quantile demand forecast (q_0.1, q_0.5, q_0.9) over the next 24 time steps."""
    quantiles = [0.1, 0.5, 0.9]
    t = np.arange(24)
    base = 35.0 + 15.0 * np.sin(2.0 * np.pi * t / 24.0)

    q_01 = (base * 0.8).tolist()
    q_05 = base.tolist()
    q_09 = (base * 1.25).tolist()

    return QuantilePredictionResponse(
        quantiles=quantiles,
        forecast={
            "q_0.1": [round(v, 2) for v in q_01],
            "q_0.5": [round(v, 2) for v in q_05],
            "q_0.9": [round(v, 2) for v in q_09],
        },
    )


@app.post("/schedule/milp", response_model=DispatchResponse, tags=["Optimization"])
def schedule_milp(request: DispatchRequest) -> DispatchResponse:
    """Computes global offline optimal MILP schedule for EV fleet charging."""
    if not request.sessions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session list cannot be empty.",
        )
    if not request.price_signals:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Price signal vector cannot be empty.",
        )

    feeder_cap = request.feeder_config.max_capacity_kw if request.feeder_config else 150.0
    baseline_load = request.feeder_config.baseline_load_kw if request.feeder_config else None
    ambient_temp = request.feeder_config.ambient_temp_c if request.feeder_config else None

    scheduler = MILPScheduler(
        feeder_capacity_kw=feeder_cap,
        dt_hours=request.dt_hours,
        battery_degradation_cost_eur_kwh=request.battery_degradation_cost_eur_kwh,
    )

    start_dt = pd.to_datetime(request.sessions[0].arrival_time)
    formatted_sessions = []

    for sess in request.sessions:
        arr_dt = pd.to_datetime(sess.arrival_time)
        dep_dt = pd.to_datetime(sess.departure_time)

        arr_step = max(0, int((arr_dt - start_dt).total_seconds() / (3600.0 * request.dt_hours)))
        dep_step = max(arr_step + 1, int((dep_dt - start_dt).total_seconds() / (3600.0 * request.dt_hours)))

        formatted_sessions.append({
            "session_id": sess.session_id,
            "charger_id": sess.charger_id,
            "arr_step": arr_step,
            "dep_step": dep_step,
            "required_energy_kwh": sess.required_energy_kwh,
            "max_charger_power_kw": sess.max_charger_power_kw,
        })

    sol = scheduler.solve(
        sessions=formatted_sessions,
        price_signal=request.price_signals,
        horizon_steps=request.horizon_steps,
        baseline_load=baseline_load,
        ambient_temp_c=ambient_temp,
    )

    power_mat = sol["power_matrix"]
    schedule_items = []

    for i, sess in enumerate(request.sessions):
        powers = [round(float(p), 2) for p in power_mat[i, :]]
        schedule_items.append(
            DispatchScheduleItem(
                session_id=sess.session_id,
                charger_id=sess.charger_id,
                power_kw=powers,
            )
        )

    return DispatchResponse(
        status=sol["status"],
        total_cost_eur=sol["total_cost_eur"],
        peak_load_kw=sol["peak_load_kw"],
        total_unmet_kwh=sol["total_unmet_kwh"],
        feeder_capacity_kw=sol.get("feeder_capacity_kw", feeder_cap),
        schedule=schedule_items,
    )


@app.post("/schedule/mpc", response_model=DispatchResponse, tags=["Optimization"])
def schedule_mpc(request: DispatchRequest) -> DispatchResponse:
    """Computes dynamic rolling-horizon MPC dispatch schedule for active EV sessions."""
    if not request.sessions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session list cannot be empty.",
        )

    feeder_cap = request.feeder_config.max_capacity_kw if request.feeder_config else 150.0
    baseline_load = request.feeder_config.baseline_load_kw if request.feeder_config else None
    ambient_temp = request.feeder_config.ambient_temp_c if request.feeder_config else None

    mpc = MPCController(
        feeder_capacity_kw=feeder_cap,
        horizon_steps=request.horizon_steps,
        dt_hours=request.dt_hours,
        battery_degradation_cost_eur_kwh=request.battery_degradation_cost_eur_kwh,
        ambient_temp_c=ambient_temp,
    )

    start_dt = pd.to_datetime(request.sessions[0].arrival_time)
    formatted_sessions = []

    for sess in request.sessions:
        arr_dt = pd.to_datetime(sess.arrival_time)
        dep_dt = pd.to_datetime(sess.departure_time)

        arr_step = max(0, int((arr_dt - start_dt).total_seconds() / (3600.0 * request.dt_hours)))
        dep_step = max(arr_step + 1, int((dep_dt - start_dt).total_seconds() / (3600.0 * request.dt_hours)))

        formatted_sessions.append({
            "session_id": sess.session_id,
            "charger_id": sess.charger_id,
            "arr_step": arr_step,
            "dep_step": dep_step,
            "initial_soc": sess.initial_soc,
            "battery_capacity_kwh": sess.battery_capacity_kwh,
            "required_energy_kwh": sess.required_energy_kwh,
            "max_charger_power_kw": sess.max_charger_power_kw,
        })

    res = mpc.run_simulation(
        sessions=formatted_sessions,
        full_price_signal=request.price_signals,
        total_steps=len(request.price_signals),
        baseline_load=baseline_load,
        ambient_temp_c=ambient_temp,
    )

    power_mat = res["dispatch_matrix"]
    schedule_items = []

    for i, sess in enumerate(request.sessions):
        powers = [round(float(p), 2) for p in power_mat[i, :]]
        schedule_items.append(
            DispatchScheduleItem(
                session_id=sess.session_id,
                charger_id=sess.charger_id,
                power_kw=powers,
            )
        )

    return DispatchResponse(
        status="OPTIMAL",
        total_cost_eur=round(res["total_cost_eur"], 2),
        peak_load_kw=round(res["peak_load_kw"], 2),
        total_unmet_kwh=round(res["unmet_energy_kwh"], 2),
        feeder_capacity_kw=feeder_cap,
        schedule=schedule_items,
    )
