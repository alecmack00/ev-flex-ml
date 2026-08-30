"""
Counterfactual backtest engine simulating Unmanaged, Time-Of-Use (TOU), and Smart MPC charging strategies.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    comfort_score,
    cost_savings_pct,
    feeder_overload_energy_kwh,
    feeder_overload_hours,
    peak_feeder_power,
    peak_reduction_pct,
    total_cost,
    unmet_energy_kwh,
)
from src.optimization.constraints import FeederConstraintManager
from src.optimization.mpc_controller import MPCController
from src.utils.logger import setup_logger

logger = setup_logger("backtest")


class BacktestEngine:
    """Simulates fleet charging execution under Unmanaged, TOU, and Smart MPC control strategies."""

    def __init__(
        self,
        feeder_capacity_kw: float = 150.0,
        dt_hours: float = 0.25,
        charging_efficiency: float = 0.95,
        tou_peak_hours: Optional[List[int]] = None,
    ) -> None:
        """Initializes BacktestEngine.

        Args:
            feeder_capacity_kw: Transformer max capacity in kW.
            dt_hours: Time resolution in hours.
            charging_efficiency: Conversion efficiency.
            tou_peak_hours: Hours designated as peak pricing for TOU strategy.
        """
        self.feeder_capacity_kw = feeder_capacity_kw
        self.dt_hours = dt_hours
        self.charging_efficiency = charging_efficiency
        self.tou_peak_hours = tou_peak_hours if tou_peak_hours is not None else [8, 9, 10, 17, 18, 19, 20]

        self.constraint_mgr = FeederConstraintManager(
            feeder_capacity_kw=feeder_capacity_kw,
            charging_efficiency=charging_efficiency,
        )

    def _prepare_sessions(
        self,
        sessions: Union[pd.DataFrame, List[Dict[str, Any]]],
        start_time: Optional[Union[str, pd.Timestamp]] = None,
    ) -> List[Dict[str, Any]]:
        """Converts raw DataFrames or timestamped dict lists into discrete step-indexed session dictionaries.

        Args:
            sessions: DataFrame or list of session dictionaries.
            start_time: Optional reference start datetime. If None, inferred from earliest arrival_time.

        Returns:
            List[Dict[str, Any]]: Formatted session dictionaries with arr_step, dep_step, etc.
        """
        if isinstance(sessions, pd.DataFrame):
            df = sessions.copy()
            if "arrival_time" in df.columns:
                df["arrival_time"] = pd.to_datetime(df["arrival_time"])
                if start_time is None:
                    start_dt = df["arrival_time"].min()
                else:
                    start_dt = pd.to_datetime(start_time)

                if "departure_time" in df.columns:
                    df["departure_time"] = pd.to_datetime(df["departure_time"])
                elif "duration_hours" in df.columns:
                    df["departure_time"] = df["arrival_time"] + pd.to_timedelta(df["duration_hours"], unit="h")

                formatted = []
                for _, row in df.iterrows():
                    arr_sec = (row["arrival_time"] - start_dt).total_seconds()
                    dep_sec = (row["departure_time"] - start_dt).total_seconds()
                    arr_step = max(0, int(round(arr_sec / (3600.0 * self.dt_hours))))
                    dep_step = max(arr_step + 1, int(round(dep_sec / (3600.0 * self.dt_hours))))

                    sess_dict = {
                        "session_id": row.get("session_id", f"SESS_{len(formatted)+1}"),
                        "arr_step": arr_step,
                        "dep_step": dep_step,
                        "required_energy_kwh": float(row.get("required_energy_kwh", 20.0)),
                        "max_charger_power_kw": float(row.get("max_charger_power_kw", 11.0)),
                        "initial_soc": float(row.get("initial_soc", 0.2)),
                        "target_soc": float(row.get("target_soc", 0.9)),
                        "battery_capacity_kwh": float(row.get("battery_capacity_kwh", 50.0)),
                    }
                    formatted.append(sess_dict)
                return formatted
            else:
                return df.to_dict(orient="records")

        elif isinstance(sessions, list):
            if len(sessions) > 0 and "arrival_time" in sessions[0] and "arr_step" not in sessions[0]:
                if start_time is None:
                    start_dt = min(pd.to_datetime(s["arrival_time"]) for s in sessions)
                else:
                    start_dt = pd.to_datetime(start_time)

                formatted = []
                for s in sessions:
                    s_copy = dict(s)
                    arr_t = pd.to_datetime(s_copy["arrival_time"])
                    if "departure_time" in s_copy:
                        dep_t = pd.to_datetime(s_copy["departure_time"])
                    elif "duration_hours" in s_copy:
                        dep_t = arr_t + pd.to_timedelta(s_copy["duration_hours"], unit="h")
                    else:
                        dep_t = arr_t + pd.to_timedelta(4.0, unit="h")

                    arr_step = max(0, int(round((arr_t - start_dt).total_seconds() / (3600.0 * self.dt_hours))))
                    dep_step = max(arr_step + 1, int(round((dep_t - start_dt).total_seconds() / (3600.0 * self.dt_hours))))

                    s_copy["arr_step"] = arr_step
                    s_copy["dep_step"] = dep_step
                    formatted.append(s_copy)
                return formatted
            return sessions

        return list(sessions)

    def run_unmanaged(
        self,
        sessions: Union[pd.DataFrame, List[Dict[str, Any]]],
        prices: np.ndarray,
        total_steps: int,
    ) -> Dict[str, Any]:
        """Simulates Unmanaged (Immediate) charging strategy.

        Args:
            sessions: List of EV charging sessions or raw DataFrame.
            prices: Price vector €/kWh per step.
            total_steps: Total simulation steps.

        Returns:
            Dict[str, Any]: Strategy simulation metrics and dispatch matrix.
        """
        prepared_sessions = self._prepare_sessions(sessions)
        N = len(prepared_sessions)
        dispatch_matrix = np.zeros((N, total_steps), dtype=np.float64)
        delivered_energy = np.zeros(N, dtype=np.float64)

        for i, sess in enumerate(prepared_sessions):
            arr_s = sess.get("arr_step", 0)
            dep_s = sess.get("dep_step", total_steps)
            req_e = sess.get("required_energy_kwh", 20.0)
            max_p = sess.get("max_charger_power_kw", 11.0)

            for t in range(arr_s, min(dep_s, total_steps)):
                rem_e = req_e - delivered_energy[i]
                if rem_e <= 1e-4:
                    break

                # Max power limited by remaining energy needed in this step
                max_p_step = rem_e / (self.dt_hours * self.charging_efficiency)
                p_act = min(max_p, max_p_step)

                dispatch_matrix[i, t] = p_act
                delivered_energy[i] += p_act * self.dt_hours * self.charging_efficiency

        feeder_load = np.sum(dispatch_matrix, axis=0)
        c_tot = total_cost(feeder_load, prices, self.dt_hours)
        p_peak = peak_feeder_power(feeder_load)
        c_score = comfort_score(delivered_energy, [s.get("required_energy_kwh", 20.0) for s in prepared_sessions])
        u_kwh = unmet_energy_kwh(delivered_energy, [s.get("required_energy_kwh", 20.0) for s in prepared_sessions])
        ov_kwh = feeder_overload_energy_kwh(feeder_load, self.feeder_capacity_kw, self.dt_hours)
        ov_hrs = feeder_overload_hours(feeder_load, self.feeder_capacity_kw, self.dt_hours)

        return {
            "strategy": "Unmanaged",
            "dispatch_matrix": dispatch_matrix,
            "feeder_load_kw": feeder_load,
            "total_cost_eur": round(c_tot, 2),
            "peak_load_kw": round(p_peak, 2),
            "overload_energy_kwh": round(ov_kwh, 2),
            "overload_hours": round(ov_hrs, 2),
            "comfort_score_pct": round(c_score, 2),
            "unmet_energy_kwh": round(u_kwh, 2),
            "delivered_energy_kwh": delivered_energy,
        }

    def run_tou(
        self,
        sessions: Union[pd.DataFrame, List[Dict[str, Any]]],
        prices: np.ndarray,
        total_steps: int,
    ) -> Dict[str, Any]:
        """Simulates Time-Of-Use (TOU) tariff-driven charging strategy.

        Args:
            sessions: List of EV charging sessions or raw DataFrame.
            prices: Price vector.
            total_steps: Total steps.

        Returns:
            Dict[str, Any]: Strategy simulation metrics and dispatch matrix.
        """
        prepared_sessions = self._prepare_sessions(sessions)
        N = len(prepared_sessions)
        dispatch_matrix = np.zeros((N, total_steps), dtype=np.float64)
        delivered_energy = np.zeros(N, dtype=np.float64)

        for i, sess in enumerate(prepared_sessions):
            arr_s = sess.get("arr_step", 0)
            dep_s = sess.get("dep_step", total_steps)
            req_e = sess.get("required_energy_kwh", 20.0)
            max_p = sess.get("max_charger_power_kw", 11.0)

            for t in range(arr_s, min(dep_s, total_steps)):
                rem_e = req_e - delivered_energy[i]
                if rem_e <= 1e-4:
                    break

                hour_of_day = int((t * self.dt_hours) % 24)
                rem_steps = dep_s - t
                max_deliverable_remaining = rem_steps * max_p * self.dt_hours * self.charging_efficiency

                # Charge if off-peak hour or if deadline requires urgent charging
                is_peak = hour_of_day in self.tou_peak_hours
                must_charge = rem_e >= 0.85 * max_deliverable_remaining

                if not is_peak or must_charge:
                    max_p_step = rem_e / (self.dt_hours * self.charging_efficiency)
                    p_act = min(max_p, max_p_step)
                    dispatch_matrix[i, t] = p_act
                    delivered_energy[i] += p_act * self.dt_hours * self.charging_efficiency

        feeder_load = np.sum(dispatch_matrix, axis=0)
        c_tot = total_cost(feeder_load, prices, self.dt_hours)
        p_peak = peak_feeder_power(feeder_load)
        c_score = comfort_score(delivered_energy, [s.get("required_energy_kwh", 20.0) for s in prepared_sessions])
        u_kwh = unmet_energy_kwh(delivered_energy, [s.get("required_energy_kwh", 20.0) for s in prepared_sessions])
        ov_kwh = feeder_overload_energy_kwh(feeder_load, self.feeder_capacity_kw, self.dt_hours)
        ov_hrs = feeder_overload_hours(feeder_load, self.feeder_capacity_kw, self.dt_hours)

        return {
            "strategy": "TOU_Tariff",
            "dispatch_matrix": dispatch_matrix,
            "feeder_load_kw": feeder_load,
            "total_cost_eur": round(c_tot, 2),
            "peak_load_kw": round(p_peak, 2),
            "overload_energy_kwh": round(ov_kwh, 2),
            "overload_hours": round(ov_hrs, 2),
            "comfort_score_pct": round(c_score, 2),
            "unmet_energy_kwh": round(u_kwh, 2),
            "delivered_energy_kwh": delivered_energy,
        }

    def run_smart_mpc(
        self,
        sessions: Union[pd.DataFrame, List[Dict[str, Any]]],
        prices: np.ndarray,
        total_steps: int,
        horizon_steps: int = 96,
    ) -> Dict[str, Any]:
        """Runs Smart MPC dynamic optimization strategy.

        Args:
            sessions: EV session specifications or raw DataFrame.
            prices: Electricity spot price schedule.
            total_steps: Total steps.
            horizon_steps: MPC lookahead horizon.

        Returns:
            Dict[str, Any]: MPC performance metrics and trace.
        """
        prepared_sessions = self._prepare_sessions(sessions)
        mpc = MPCController(
            feeder_capacity_kw=self.feeder_capacity_kw,
            horizon_steps=horizon_steps,
            dt_hours=self.dt_hours,
            charging_efficiency=self.charging_efficiency,
        )

        res = mpc.run_simulation(prepared_sessions, prices, total_steps)
        delivered_energy = res["delivered_energy_kwh"]
        feeder_load = res["total_feeder_load_kw"]

        c_tot = res["total_cost_eur"]
        p_peak = res["peak_load_kw"]
        c_score = comfort_score(delivered_energy, [s.get("required_energy_kwh", 20.0) for s in prepared_sessions])
        u_kwh = res["unmet_energy_kwh"]
        ov_kwh = feeder_overload_energy_kwh(feeder_load, self.feeder_capacity_kw, self.dt_hours)
        ov_hrs = feeder_overload_hours(feeder_load, self.feeder_capacity_kw, self.dt_hours)

        return {
            "strategy": "Smart_MPC",
            "dispatch_matrix": res["dispatch_matrix"],
            "feeder_load_kw": feeder_load,
            "total_cost_eur": round(c_tot, 2),
            "peak_load_kw": round(p_peak, 2),
            "overload_energy_kwh": round(ov_kwh, 2),
            "overload_hours": round(ov_hrs, 2),
            "comfort_score_pct": round(c_score, 2),
            "unmet_energy_kwh": round(u_kwh, 2),
            "delivered_energy_kwh": delivered_energy,
        }

    def run_backtest_comparison(
        self,
        sessions: Union[pd.DataFrame, List[Dict[str, Any]]],
        price_signal: Union[List[float], np.ndarray],
        total_steps: int = 192,  # 48 hours
        start_time: Optional[Union[str, pd.Timestamp]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """Runs comparative side-by-side backtest of Unmanaged vs TOU vs Smart MPC.

        Args:
            sessions: Session specifications list or raw DataFrame.
            price_signal: Price schedule array.
            total_steps: Number of evaluation time steps.
            start_time: Optional start timestamp for simulation alignment.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]: Performance comparison DataFrame and full detailed results dictionary.
        """
        prices = np.array(price_signal, dtype=np.float64)
        prepared_sessions = self._prepare_sessions(sessions, start_time=start_time)

        logger.info(f"Running counterfactual backtest across 3 strategies on {len(prepared_sessions)} EV sessions.")

        res_unmanaged = self.run_unmanaged(prepared_sessions, prices, total_steps)
        res_tou = self.run_tou(prepared_sessions, prices, total_steps)
        res_smart = self.run_smart_mpc(prepared_sessions, prices, total_steps)

        base_cost = res_unmanaged["total_cost_eur"]
        base_peak = res_unmanaged["peak_load_kw"]

        comparison_data = [
            {
                "Strategy": res_unmanaged["strategy"],
                "Total Cost (€)": res_unmanaged["total_cost_eur"],
                "Cost Savings (%)": 0.0,
                "Peak Feeder Load (kW)": res_unmanaged["peak_load_kw"],
                "Peak Shaving (%)": 0.0,
                "Overload Energy (kWh)": res_unmanaged["overload_energy_kwh"],
                "Overload Duration (h)": res_unmanaged["overload_hours"],
                "Comfort Score (%)": res_unmanaged["comfort_score_pct"],
                "Unmet Energy (kWh)": res_unmanaged["unmet_energy_kwh"],
            },
            {
                "Strategy": res_tou["strategy"],
                "Total Cost (€)": res_tou["total_cost_eur"],
                "Cost Savings (%)": round(cost_savings_pct(res_tou["total_cost_eur"], base_cost), 2),
                "Peak Feeder Load (kW)": res_tou["peak_load_kw"],
                "Peak Shaving (%)": round(peak_reduction_pct(res_tou["peak_load_kw"], base_peak), 2),
                "Overload Energy (kWh)": res_tou["overload_energy_kwh"],
                "Overload Duration (h)": res_tou["overload_hours"],
                "Comfort Score (%)": res_tou["comfort_score_pct"],
                "Unmet Energy (kWh)": res_tou["unmet_energy_kwh"],
            },
            {
                "Strategy": res_smart["strategy"],
                "Total Cost (€)": res_smart["total_cost_eur"],
                "Cost Savings (%)": round(cost_savings_pct(res_smart["total_cost_eur"], base_cost), 2),
                "Peak Feeder Load (kW)": res_smart["peak_load_kw"],
                "Peak Shaving (%)": round(peak_reduction_pct(res_smart["peak_load_kw"], base_peak), 2),
                "Overload Energy (kWh)": res_smart["overload_energy_kwh"],
                "Overload Duration (h)": res_smart["overload_hours"],
                "Comfort Score (%)": res_smart["comfort_score_pct"],
                "Unmet Energy (kWh)": res_smart["unmet_energy_kwh"],
            },
        ]

        df_summary = pd.DataFrame(comparison_data)

        raw_results = {
            "Unmanaged": res_unmanaged,
            "TOU": res_tou,
            "Smart_MPC": res_smart,
        }

        return df_summary, raw_results
