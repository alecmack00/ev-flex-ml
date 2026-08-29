"""
Evaluation metrics for cost savings, peak feeder load shaving, and driver SLA comfort scores.
"""

from typing import Dict, List, Union

import numpy as np


def total_cost(
    power_schedule: np.ndarray,
    prices_eur_kwh: np.ndarray,
    dt_hours: float = 0.25,
) -> float:
    """Calculates total electricity procurement cost (€).

    Args:
        power_schedule: Power dispatch matrix [num_sessions, num_steps] or aggregate array [num_steps].
        prices_eur_kwh: Pricing vector [num_steps] in €/kWh.
        dt_hours: Time resolution in hours.

    Returns:
        float: Total cost in EUR.
    """
    if power_schedule.ndim == 2:
        feeder_load = np.sum(power_schedule, axis=0)
    else:
        feeder_load = power_schedule

    steps = min(len(feeder_load), len(prices_eur_kwh))
    cost = np.sum(feeder_load[:steps] * prices_eur_kwh[:steps] * dt_hours)
    return float(cost)


def cost_savings_pct(smart_cost: float, baseline_cost: float) -> float:
    """Computes relative percentage cost savings over baseline charging strategy.

    Args:
        smart_cost: Cost under optimized flexibility strategy (€).
        baseline_cost: Cost under baseline strategy (€).

    Returns:
        float: Percentage savings (e.g. 24.5 for 24.5% reduction).
    """
    if baseline_cost <= 0:
        return 0.0
    savings = (baseline_cost - smart_cost) / baseline_cost * 100.0
    return float(savings)


def peak_feeder_power(power_schedule: np.ndarray) -> float:
    """Finds maximum aggregate power draw across the feeder transformer (kW).

    Args:
        power_schedule: Dispatch matrix or 1D aggregate feeder load array.

    Returns:
        float: Peak load in kW.
    """
    if power_schedule.ndim == 2:
        feeder_load = np.sum(power_schedule, axis=0)
    else:
        feeder_load = power_schedule

    if len(feeder_load) == 0:
        return 0.0
    return float(np.max(feeder_load))


def peak_reduction_pct(smart_peak: float, baseline_peak: float) -> float:
    """Computes percentage reduction in feeder peak demand (peak shaving).

    Args:
        smart_peak: Peak load under optimized strategy (kW).
        baseline_peak: Peak load under baseline strategy (kW).

    Returns:
        float: Percentage peak load reduction.
    """
    if baseline_peak <= 0:
        return 0.0
    reduction = (baseline_peak - smart_peak) / baseline_peak * 100.0
    return float(reduction)


def comfort_score(
    delivered_energy: Union[List[float], np.ndarray],
    required_energy: Union[List[float], np.ndarray],
) -> float:
    """Calculates SLA comfort score percentage (% of requested energy successfully delivered at departure).

    Comfort Score = (Sum(min(E_delivered, E_required)) / Sum(E_required)) * 100

    Args:
        delivered_energy: Array of delivered energy per session (kWh).
        required_energy: Array of required energy per session (kWh).

    Returns:
        float: Comfort score percentage [0.0, 100.0].
    """
    del_arr = np.array(delivered_energy, dtype=np.float64)
    req_arr = np.array(required_energy, dtype=np.float64)

    total_req = np.sum(req_arr)
    if total_req <= 0:
        return 100.0

    satisfied_energy = np.sum(np.minimum(del_arr, req_arr))
    score = (satisfied_energy / total_req) * 100.0
    return float(score)


def unmet_energy_kwh(
    delivered_energy: Union[List[float], np.ndarray],
    required_energy: Union[List[float], np.ndarray],
) -> float:
    """Computes total deficit in energy at driver departure deadline across all fleet sessions.

    Args:
        delivered_energy: Array of delivered energy per session (kWh).
        required_energy: Array of required energy per session (kWh).

    Returns:
        float: Total unmet energy in kWh.
    """
    del_arr = np.array(delivered_energy, dtype=np.float64)
    req_arr = np.array(required_energy, dtype=np.float64)

    unmet = np.sum(np.maximum(0.0, req_arr - del_arr))
    return float(unmet)


def feeder_overload_energy_kwh(
    power_schedule: np.ndarray,
    feeder_capacity_kw: float,
    dt_hours: float = 0.25,
) -> float:
    """Calculates total energy drawn in excess of transformer capacity rating (kWh).

    Args:
        power_schedule: Power dispatch matrix [num_sessions, num_steps] or aggregate load array [num_steps].
        feeder_capacity_kw: Maximum transformer capacity rating in kW.
        dt_hours: Time resolution in hours.

    Returns:
        float: Total overload energy in kWh.
    """
    if power_schedule.ndim == 2:
        feeder_load = np.sum(power_schedule, axis=0)
    else:
        feeder_load = power_schedule

    overload_power = np.maximum(0.0, feeder_load - feeder_capacity_kw)
    total_overload_kwh = np.sum(overload_power) * dt_hours
    return float(total_overload_kwh)


def feeder_overload_hours(
    power_schedule: np.ndarray,
    feeder_capacity_kw: float,
    dt_hours: float = 0.25,
) -> float:
    """Calculates total duration (hours) where transformer capacity rating was exceeded.

    Args:
        power_schedule: Power dispatch matrix [num_sessions, num_steps] or aggregate load array [num_steps].
        feeder_capacity_kw: Maximum transformer capacity rating in kW.
        dt_hours: Time resolution in hours.

    Returns:
        float: Total overload duration in hours.
    """
    if power_schedule.ndim == 2:
        feeder_load = np.sum(power_schedule, axis=0)
    else:
        feeder_load = power_schedule

    overload_steps = np.sum(feeder_load > feeder_capacity_kw + 1e-4)
    return float(overload_steps * dt_hours)
