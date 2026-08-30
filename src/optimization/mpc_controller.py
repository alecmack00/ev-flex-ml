"""
Rolling-horizon Model Predictive Controller (MPC) for dynamic EV fleet charging under uncertainty.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.optimization.constraints import FeederConstraintManager
from src.optimization.milp_scheduler import MILPScheduler
from src.utils.logger import setup_logger

logger = setup_logger("mpc_controller")


class MPCController:
    """Dynamic rolling-horizon Model Predictive Controller executing real-time dispatch decisions."""

    def __init__(
        self,
        feeder_capacity_kw: float = 150.0,
        horizon_steps: int = 96,  # 24 hours at 15-minute resolution
        dt_hours: float = 0.25,
        charging_efficiency: float = 0.95,
        ml_model: Optional[Any] = None,
    ) -> None:
        """Initializes MPCController.

        Args:
            feeder_capacity_kw: Transformer limit in kW.
            horizon_steps: MPC lookahead horizon H (number of time steps).
            dt_hours: Time resolution in hours (0.25 for 15 min).
            charging_efficiency: Charger conversion efficiency.
            ml_model: Optional probabilistic forecasting model (MDN or TCN).
        """
        self.feeder_capacity_kw = feeder_capacity_kw
        self.horizon_steps = horizon_steps
        self.dt_hours = dt_hours
        self.charging_efficiency = charging_efficiency
        self.ml_model = ml_model

        self.scheduler = MILPScheduler(
            feeder_capacity_kw=feeder_capacity_kw,
            charging_efficiency=charging_efficiency,
            dt_hours=dt_hours,
        )
        self.constraint_mgr = FeederConstraintManager(
            feeder_capacity_kw=feeder_capacity_kw,
            charging_efficiency=charging_efficiency,
        )

    def predict_session_params(
        self,
        session: Dict[str, Any],
        current_step: int,
        preprocessor: Optional[Any] = None,
    ) -> Tuple[int, float]:
        """Estimates departure step and required energy for a session using driver inputs or ML model fallback.

        Args:
            session: Session specification dictionary.
            current_step: Current simulation step index k.
            preprocessor: Optional SessionPreprocessor instance.

        Returns:
            Tuple[int, float]: (Estimated departure step index, Estimated required energy in kWh).
        """
        has_dep = "dep_step" in session and session["dep_step"] is not None
        has_energy = "required_energy_kwh" in session and session["required_energy_kwh"] is not None

        # 1. Use explicit driver inputs if available
        if has_dep and has_energy:
            return int(session["dep_step"]), float(session["required_energy_kwh"])

        # 2. Deep Learning Model Fallback (MDN forecasting)
        if self.ml_model is not None:
            try:
                import torch
                row = {
                    "arrival_time": session.get("arrival_time", pd.Timestamp.now()),
                    "battery_capacity_kwh": session.get("battery_capacity_kwh", 60.0),
                    "initial_soc": session.get("initial_soc", 0.2),
                    "target_soc": session.get("target_soc", 0.9),
                    "required_energy_kwh": session.get("required_energy_kwh", 35.0),
                    "max_charger_power_kw": session.get("max_charger_power_kw", 11.0),
                }
                df = pd.DataFrame([row])

                if preprocessor is None:
                    from src.data.preprocessor import SessionPreprocessor
                    preprocessor = SessionPreprocessor()
                    preprocessor.fit_transform(df)

                X_scaled, _ = preprocessor.transform(df)
                x_tensor = torch.tensor(X_scaled, dtype=torch.float32)

                if hasattr(self.ml_model, "predict_distribution"):
                    mean_pred, _ = self.ml_model.predict_distribution(x_tensor)
                    pred_dur = max(1.0, float(mean_pred[0, 0].item()))
                    pred_req_e = max(1.0, float(mean_pred[0, 1].item()))
                else:
                    pred_dur, pred_req_e = 4.0, 25.0

                dep_s = current_step + max(1, int(round(pred_dur / self.dt_hours)))
                req_e = float(session["required_energy_kwh"]) if has_energy else pred_req_e
                return int(dep_s), float(req_e)
            except Exception as e:
                logger.warning(f"ML model inference fallback failed: {e}. Using heuristic default.")

        # 3. Default Heuristic Fallback
        dep_s = session.get("dep_step", current_step + int(4.0 / self.dt_hours))
        req_e = session.get("required_energy_kwh", 20.0)
        return int(dep_s), float(req_e)

    def run_simulation(
        self,
        sessions: List[Dict[str, Any]],
        full_price_signal: Union[List[float], np.ndarray],
        total_steps: int,
    ) -> Dict[str, Any]:
        """Executes full rolling-horizon MPC simulation over total_steps time series.

        Args:
            sessions: List of EV charging session specification dicts.
            full_price_signal: Array of prices for all steps t in [0, total_steps + horizon_steps].
            total_steps: Number of simulation steps to run.

        Returns:
            Dict[str, Any]: Simulation trace containing dispatch matrix, battery SoC trajectories, and costs.
        """
        prices = np.array(full_price_signal, dtype=np.float64)
        N = len(sessions)

        # Output dispatch matrix [N, total_steps]
        dispatch_matrix = np.zeros((N, total_steps), dtype=np.float64)
        delivered_energy_kwh = np.zeros(N, dtype=np.float64)

        # Track battery SoC over time [N, total_steps + 1]
        soc_history = np.zeros((N, total_steps + 1), dtype=np.float64)
        for i, sess in enumerate(sessions):
            soc_history[i, 0] = sess.get("initial_soc", 0.2)

        logger.info(f"Starting MPC rolling horizon simulation over {total_steps} steps (Horizon H={self.horizon_steps}).")

        for k in range(total_steps):
            # Price signal vector for current horizon window [k, k + H]
            p_window = prices[k : k + self.horizon_steps]
            if len(p_window) < self.horizon_steps:
                # Pad price window if near end of simulation
                pad_val = p_window[-1] if len(p_window) > 0 else 0.2
                p_window = np.pad(p_window, (0, self.horizon_steps - len(p_window)), mode="edge")

            # Active sessions at time step k
            local_sessions = []
            local_session_indices = []

            for i, sess in enumerate(sessions):
                arr_s = sess.get("arr_step", 0)
                dep_s, req_e = self.predict_session_params(sess, current_step=k)

                # Check if session is currently connected or arriving in future horizon
                if arr_s <= k < dep_s or (arr_s > k and arr_s < k + self.horizon_steps):
                    rem_e = max(0.0, req_e - delivered_energy_kwh[i])

                    # Compute relative arrival and departure steps in horizon frame [0, H]
                    rel_arr = max(0, arr_s - k)
                    rel_dep = min(self.horizon_steps, dep_s - k)

                    if rem_e > 1e-4 and rel_arr < rel_dep:
                        local_sessions.append({
                            "session_id": sess.get("session_id", f"SESS_{i}"),
                            "arr_step": rel_arr,
                            "dep_step": rel_dep,
                            "required_energy_kwh": rem_e,
                            "max_charger_power_kw": sess.get("max_charger_power_kw", 11.0),
                        })
                        local_session_indices.append(i)

            # Solve horizon optimization
            if len(local_sessions) > 0:
                sol = self.scheduler.solve(
                    local_sessions,
                    p_window,
                    horizon_steps=self.horizon_steps,
                )
                pow_mat = sol["power_matrix"]

                # Extract first step decision (t = 0 in horizon window)
                for loc_idx, glob_idx in enumerate(local_session_indices):
                    p_action = pow_mat[loc_idx, 0]
                    dispatch_matrix[glob_idx, k] = p_action
                    delivered_energy_kwh[glob_idx] += p_action * self.dt_hours * self.charging_efficiency

            # State transition update for step k+1
            for i, sess in enumerate(sessions):
                p_act = dispatch_matrix[i, k]
                bat_cap = sess.get("battery_capacity_kwh", 50.0)
                soc_history[i, k + 1] = self.constraint_mgr.compute_soc_step(
                    current_soc=soc_history[i, k],
                    power_kw=p_act,
                    battery_capacity_kwh=bat_cap,
                    dt_hours=self.dt_hours,
                )

        # Performance summary metrics
        total_feeder_load = np.sum(dispatch_matrix, axis=0)
        peak_load = float(np.max(total_feeder_load)) if len(total_feeder_load) > 0 else 0.0

        step_costs = total_feeder_load * prices[:total_steps] * self.dt_hours
        total_cost = float(np.sum(step_costs))

        unmet_energy_total = 0.0
        for i, sess in enumerate(sessions):
            req_e = sess.get("required_energy_kwh", 20.0)
            unmet = max(0.0, req_e - delivered_energy_kwh[i])
            unmet_energy_total += unmet

        logger.info(
            f"MPC Simulation Complete. Total Cost: {total_cost:.2f} EUR | Peak Load: {peak_load:.2f} kW | Unmet: {unmet_energy_total:.2f} kWh"
        )

        return {
            "dispatch_matrix": dispatch_matrix,
            "soc_history": soc_history,
            "total_feeder_load_kw": total_feeder_load,
            "peak_load_kw": peak_load,
            "total_cost_eur": total_cost,
            "delivered_energy_kwh": delivered_energy_kwh,
            "unmet_energy_kwh": unmet_energy_total,
        }
