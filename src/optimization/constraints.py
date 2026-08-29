"""
Feeder transformer limits, EV battery state-of-charge tracking, and charger hardware boundary constraints.
"""

from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger("constraints")


class FeederConstraintManager:
    """Manages system operational limits, battery SoC update equations, and solver boundary conditions."""

    def __init__(
        self,
        feeder_capacity_kw: float = 150.0,
        safety_margin: float = 0.90,
        charging_efficiency: float = 0.95,
        default_max_charger_kw: float = 11.0,
    ) -> None:
        """Initializes FeederConstraintManager.

        Args:
            feeder_capacity_kw: Transformer max capacity in kW.
            safety_margin: Feeder operational safety margin multiplier (e.g. 0.90 for 10% headroom).
            charging_efficiency: Charger AC-to-DC conversion efficiency (eta).
            default_max_charger_kw: Default max power draw per charger point in kW.
        """
        self.feeder_capacity_kw = feeder_capacity_kw
        self.safety_margin = safety_margin
        self.effective_feeder_limit_kw = feeder_capacity_kw * safety_margin
        self.charging_efficiency = charging_efficiency
        self.default_max_charger_kw = default_max_charger_kw

    def compute_soc_step(
        self,
        current_soc: float,
        power_kw: float,
        battery_capacity_kwh: float,
        dt_hours: float = 0.25,
    ) -> float:
        """Computes next State of Charge (SoC) step for a single vehicle.

        SoC(t+1) = SoC(t) + (eta * P(t) * dt) / E_cap

        Args:
            current_soc: Current SoC in fraction [0.0, 1.0].
            power_kw: Charging power in kW during current interval.
            battery_capacity_kwh: Total battery pack capacity in kWh.
            dt_hours: Time step duration in hours (e.g. 0.25h for 15 min).

        Returns:
            float: Updated SoC fraction [0.0, 1.0].
        """
        energy_delivered = self.charging_efficiency * power_kw * dt_hours
        delta_soc = energy_delivered / battery_capacity_kwh
        new_soc = float(np.clip(current_soc + delta_soc, 0.0, 1.0))
        return new_soc

    def validate_schedule(
        self,
        power_schedule_matrix: np.ndarray,
        sessions: List[Dict[str, Any]],
        dt_hours: float = 0.25,
    ) -> Dict[str, Any]:
        """Validates that a proposed power schedule satisfies feeder and battery physical limits.

        Args:
            power_schedule_matrix: 2D numpy array [num_sessions, num_time_steps] of charging powers (kW).
            sessions: List of session specification dicts.
            dt_hours: Time step duration in hours.

        Returns:
            Dict[str, Any]: Validation report with boolean status and detailed violations.
        """
        num_sessions, num_steps = power_schedule_matrix.shape
        violations = []

        # 1. Check feeder transformer capacity limits per time step
        total_feeder_power = np.sum(power_schedule_matrix, axis=0)
        max_feeder_load = np.max(total_feeder_power)

        feeder_overloads = np.where(total_feeder_power > self.effective_feeder_limit_kw + 1e-3)[0]
        if len(feeder_overloads) > 0:
            for step in feeder_overloads:
                violations.append({
                    "type": "FEEDER_CAPACITY_EXCEEDED",
                    "step": int(step),
                    "power_kw": float(total_feeder_power[step]),
                    "limit_kw": self.effective_feeder_limit_kw,
                })

        # 2. Check individual session constraints
        unmet_energy_total = 0.0

        for i, sess in enumerate(sessions):
            sess_power = power_schedule_matrix[i, :]
            arr_step = sess.get("arr_step", 0)
            dep_step = sess.get("dep_step", num_steps)
            req_energy = sess.get("required_energy_kwh", 20.0)
            max_p = sess.get("max_charger_power_kw", self.default_max_charger_kw)

            # Non-plugin power check
            out_of_bounds_power = np.sum(sess_power[:arr_step]) + np.sum(sess_power[dep_step:])
            if out_of_bounds_power > 1e-3:
                violations.append({
                    "type": "POWER_DRAWN_WHILE_UNPLUGGED",
                    "session_id": sess.get("session_id", f"SESS_{i}"),
                    "power_kw": float(out_of_bounds_power),
                })

            # Max charger power check
            if np.max(sess_power) > max_p + 1e-3:
                violations.append({
                    "type": "CHARGER_POWER_EXCEEDED",
                    "session_id": sess.get("session_id", f"SESS_{i}"),
                    "max_draw_kw": float(np.max(sess_power)),
                    "charger_limit_kw": max_p,
                })

            # Energy target check
            delivered_energy = np.sum(sess_power[arr_step:dep_step]) * dt_hours * self.charging_efficiency
            if delivered_energy < req_energy - 1e-2:
                unmet = req_energy - delivered_energy
                unmet_energy_total += unmet
                violations.append({
                    "type": "UNMET_ENERGY_AT_DEPARTURE",
                    "session_id": sess.get("session_id", f"SESS_{i}"),
                    "delivered_kwh": float(delivered_energy),
                    "required_kwh": float(req_energy),
                    "unmet_kwh": float(unmet),
                })

        is_valid = len(violations) == 0
        return {
            "is_valid": is_valid,
            "max_feeder_load_kw": float(max_feeder_load),
            "feeder_limit_kw": self.effective_feeder_limit_kw,
            "total_unmet_energy_kwh": float(unmet_energy_total),
            "violation_count": len(violations),
            "violations": violations,
        }
