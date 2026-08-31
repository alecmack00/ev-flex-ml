"""
Mixed-Integer Linear Program (MILP) & Linear Program (LP) global offline fleet charging scheduler.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.optimization.constraints import FeederConstraintManager
from src.utils.logger import setup_logger

logger = setup_logger("milp_scheduler")

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False


class MILPScheduler:
    """Solves static global optimization for EV fleet charging over a full planning horizon."""

    def __init__(
        self,
        feeder_capacity_kw: float = 150.0,
        charging_efficiency: float = 0.95,
        dt_hours: float = 0.25,
        slack_penalty: float = 10.0,
        peak_penalty: float = 2.5,
        battery_degradation_cost_eur_kwh: float = 0.0,
        backend: str = "auto",
    ) -> None:
        """Initializes MILPScheduler.

        Args:
            feeder_capacity_kw: Feeder transformer maximum power limit in kW.
            charging_efficiency: Charger conversion efficiency (eta).
            dt_hours: Time resolution in hours (0.25 for 15 min).
            slack_penalty: Penalty per unmet kWh at departure (€/kWh).
            peak_penalty: Penalty per peak feeder demand kW (€/kW).
            battery_degradation_cost_eur_kwh: Battery cycling degradation penalty in €/kWh throughput.
            backend: Solver framework ('cvxpy', 'pulp', or 'auto').
        """
        self.feeder_capacity_kw = feeder_capacity_kw
        self.charging_efficiency = charging_efficiency
        self.dt_hours = dt_hours
        self.slack_penalty = slack_penalty
        self.peak_penalty = peak_penalty
        self.battery_degradation_cost_eur_kwh = max(0.0, float(battery_degradation_cost_eur_kwh))

        if backend == "auto":
            if HAS_CVXPY:
                self.backend = "cvxpy"
            elif HAS_PULP:
                self.backend = "pulp"
            else:
                raise RuntimeError("Neither CVXPY nor PuLP optimization packages are installed.")
        else:
            self.backend = backend.lower()

    def solve(
        self,
        sessions: List[Dict[str, Any]],
        price_signal: Union[List[float], np.ndarray],
        horizon_steps: Optional[int] = None,
        baseline_load: Optional[Union[List[float], np.ndarray]] = None,
        ambient_temp_c: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Solves optimal charging schedule for all EV sessions across specified price horizon.

        Args:
            sessions: List of session specification dicts containing:
                      - session_id, arr_step, dep_step, required_energy_kwh, max_charger_power_kw.
            price_signal: 1D array of spot electricity prices in EUR/kWh per step t.
            horizon_steps: Optional total steps T. Inferred from price_signal length if None.
            baseline_load: Optional 1D array of non-EV background demand (kW) consuming feeder headroom.
            ambient_temp_c: Optional ambient temperature (°C) for dynamic line/transformer thermal derating.

        Returns:
            Dict[str, Any]: Dictionary containing solution status, optimal power matrix (kW), total cost, peak load, and metrics.
        """
        prices = np.array(price_signal, dtype=np.float64)
        T = len(prices) if horizon_steps is None else horizon_steps
        prices = prices[:T]

        eff_feeder_cap = self.feeder_capacity_kw
        if ambient_temp_c is not None:
            eff_feeder_cap = FeederConstraintManager.compute_dynamic_transformer_rating(
                self.feeder_capacity_kw, ambient_temp_c
            )

        base_load_arr = None
        if baseline_load is not None:
            base_load_arr = np.asarray(baseline_load, dtype=np.float64)[:T]
            if len(base_load_arr) < T:
                base_load_arr = np.pad(base_load_arr, (0, T - len(base_load_arr)), mode="edge")

        N = len(sessions)
        if N == 0:
            return {
                "status": "OPTIMAL",
                "power_matrix": np.zeros((0, T)),
                "total_cost_eur": 0.0,
                "peak_load_kw": 0.0,
                "total_unmet_kwh": 0.0,
                "feeder_capacity_kw": eff_feeder_cap,
            }

        logger.debug(f"Solving MILP schedule for {N} sessions over {T} time steps using backend='{self.backend}'.")

        if self.backend == "cvxpy" and HAS_CVXPY:
            return self._solve_cvxpy(sessions, prices, N, T, eff_feeder_cap, base_load_arr)
        elif HAS_PULP:
            return self._solve_pulp(sessions, prices, N, T, eff_feeder_cap, base_load_arr)
        else:
            raise RuntimeError(f"Requested optimization backend '{self.backend}' is unavailable.")

    def _solve_cvxpy(
        self,
        sessions: List[Dict[str, Any]],
        prices: np.ndarray,
        N: int,
        T: int,
        feeder_cap: float,
        baseline_load: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """CVXPY solver implementation."""
        P = cp.Variable((N, T), nonneg=True, name="Power")
        slack = cp.Variable(N, nonneg=True, name="Slack")
        peak = cp.Variable(nonneg=True, name="PeakLoad")

        constraints = []

        # 1. Feeder Capacity Limit & Peak load tracking for each step t (incorporating non-EV baseline)
        for t in range(T):
            base_t = float(baseline_load[t]) if baseline_load is not None else 0.0
            total_t = cp.sum(P[:, t])
            constraints.append(total_t + base_t <= feeder_cap)
            constraints.append(total_t + base_t <= peak)

        # 2. Session arrival, departure, charger max power, and energy satisfaction
        for i, sess in enumerate(sessions):
            arr_step = max(0, int(sess.get("arr_step", 0)))
            dep_step = min(T, int(sess.get("dep_step", T)))
            req_energy = float(sess.get("required_energy_kwh", 20.0))
            max_p = float(sess.get("max_charger_power_kw", 11.0))

            # Non-plugin zero power bounds
            if arr_step > 0:
                constraints.append(P[i, :arr_step] == 0)
            if dep_step < T:
                constraints.append(P[i, dep_step:] == 0)

            # Charger maximum rate bound during active window
            if arr_step < dep_step:
                constraints.append(P[i, arr_step:dep_step] <= max_p)

            # Delivered energy target + slack >= required_energy
            delivered_energy = cp.sum(P[i, arr_step:dep_step]) * self.dt_hours * self.charging_efficiency
            constraints.append(delivered_energy + slack[i] >= req_energy)

        # Objective Function
        # Energy cost + Battery degradation (linear throughput + quadratic C-rate penalty) + Slack penalty + Peak demand charge
        energy_cost = cp.sum(cp.matmul(P, prices) * self.dt_hours)
        if self.battery_degradation_cost_eur_kwh > 0.0:
            deg_cost = self.battery_degradation_cost_eur_kwh * cp.sum(P) * self.dt_hours + \
                       0.001 * self.battery_degradation_cost_eur_kwh * cp.sum(cp.square(P)) * self.dt_hours
        else:
            deg_cost = 0.0
            
        total_slack_cost = self.slack_penalty * cp.sum(slack)
        peak_cost = self.peak_penalty * peak

        objective = cp.Minimize(energy_cost + deg_cost + total_slack_cost + peak_cost)
        problem = cp.Problem(objective, constraints)

        # Dynamic solver selection prioritizing modern solvers (CLARABEL -> HIGHS -> ECOS -> OSQP)
        installed = cp.installed_solvers()
        candidate_solvers = []
        for s in [getattr(cp, "CLARABEL", "CLARABEL"), getattr(cp, "HIGHS", "HIGHS"), getattr(cp, "ECOS", "ECOS"), getattr(cp, "OSQP", "OSQP")]:
            if s in installed and s not in candidate_solvers:
                candidate_solvers.append(s)

        solved = False
        for s in candidate_solvers:
            try:
                solver_kwargs = {"verbose": False}
                if s == getattr(cp, "CLARABEL", "CLARABEL"):
                    solver_kwargs.update({"tol_gap_abs": 1e-6, "tol_gap_rel": 1e-6, "tol_feas": 1e-6})
                elif s == getattr(cp, "OSQP", "OSQP"):
                    solver_kwargs.update({"eps_abs": 1e-5, "eps_rel": 1e-5, "max_iter": 10000})
                problem.solve(solver=s, **solver_kwargs)
                if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    solved = True
                    break
            except Exception:
                continue

        if not solved:
            try:
                problem.solve(verbose=False)
            except Exception as e:
                logger.warning(f"CVXPY default solver failed: {e}")

        status = problem.status
        power_matrix = P.value if P.value is not None else np.zeros((N, T))
        power_matrix = np.maximum(0.0, power_matrix)

        total_cost = float(energy_cost.value) if energy_cost.value is not None else 0.0
        peak_val = float(peak.value) if peak.value is not None else 0.0
        unmet_val = float(np.sum(slack.value)) if slack.value is not None else 0.0

        return {
            "status": status,
            "power_matrix": power_matrix,
            "total_cost_eur": round(total_cost, 4),
            "peak_load_kw": round(peak_val, 2),
            "total_unmet_kwh": round(unmet_val, 4),
            "feeder_capacity_kw": feeder_cap,
        }

    def _solve_pulp(
        self,
        sessions: List[Dict[str, Any]],
        prices: np.ndarray,
        N: int,
        T: int,
        feeder_cap: float,
        baseline_load: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """PuLP solver fallback implementation."""
        prob = pulp.LpProblem("EV_Fleet_Charging_Optimization", pulp.LpMinimize)

        # Variables
        P_vars = {}
        for i in range(N):
            sess = sessions[i]
            arr_step = max(0, int(sess.get("arr_step", 0)))
            dep_step = min(T, int(sess.get("dep_step", T)))
            max_p = float(sess.get("max_charger_power_kw", 11.0))

            for t in range(T):
                if arr_step <= t < dep_step:
                    P_vars[(i, t)] = pulp.LpVariable(f"P_{i}_{t}", lowBound=0.0, upBound=max_p)
                else:
                    P_vars[(i, t)] = pulp.LpVariable(f"P_{i}_{t}", lowBound=0.0, upBound=0.0)

        slack_vars = [pulp.LpVariable(f"Slack_{i}", lowBound=0.0) for i in range(N)]
        peak_var = pulp.LpVariable("PeakLoad", lowBound=0.0)

        # Objective Function
        # Energy cost + Degradation cost + Slack penalty + Peak demand charge
        effective_unit_cost = prices + self.battery_degradation_cost_eur_kwh
        energy_cost_expr = pulp.lpSum([
            P_vars[(i, t)] * effective_unit_cost[t] * self.dt_hours
            for i in range(N) for t in range(T)
        ])
        slack_cost_expr = self.slack_penalty * pulp.lpSum(slack_vars)
        peak_cost_expr = self.peak_penalty * peak_var

        prob += energy_cost_expr + slack_cost_expr + peak_cost_expr

        # Constraints (incorporating non-EV baseline load)
        for t in range(T):
            base_t = float(baseline_load[t]) if baseline_load is not None else 0.0
            step_load = pulp.lpSum([P_vars[(i, t)] for i in range(N)]) + base_t
            prob += (step_load <= feeder_cap, f"FeederCap_{t}")
            prob += (step_load <= peak_var, f"PeakLoad_{t}")

        for i, sess in enumerate(sessions):
            arr_step = max(0, int(sess.get("arr_step", 0)))
            dep_step = min(T, int(sess.get("dep_step", T)))
            req_energy = float(sess.get("required_energy_kwh", 20.0))

            energy_delivered = pulp.lpSum([
                P_vars[(i, t)] * self.dt_hours * self.charging_efficiency
                for t in range(arr_step, dep_step)
            ])
            prob += (energy_delivered + slack_vars[i] >= req_energy, f"EnergyReq_{i}")

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        power_matrix = np.zeros((N, T))
        for i in range(N):
            for t in range(T):
                val = pulp.value(P_vars[(i, t)])
                power_matrix[i, t] = max(0.0, val if val is not None else 0.0)

        status_str = pulp.LpStatus[prob.status]
        total_cost = float(pulp.value(energy_cost_expr)) if pulp.value(energy_cost_expr) is not None else 0.0
        peak_val = float(pulp.value(peak_var)) if pulp.value(peak_var) is not None else 0.0
        unmet_val = float(sum(pulp.value(s) for s in slack_vars if pulp.value(s) is not None))

        return {
            "status": status_str,
            "power_matrix": power_matrix,
            "total_cost_eur": round(total_cost, 4),
            "peak_load_kw": round(peak_val, 2),
            "total_unmet_kwh": round(unmet_val, 4),
            "feeder_capacity_kw": feeder_cap,
        }
