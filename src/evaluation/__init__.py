"""
Evaluation, counterfactual backtesting, and metric computation package.
"""

from .backtest import BacktestEngine
from .metrics import (
    comfort_score,
    cost_savings_pct,
    feeder_overload_energy_kwh,
    feeder_overload_hours,
    peak_feeder_power,
    peak_reduction_pct,
    total_cost,
    unmet_energy_kwh,
)

__all__ = [
    "BacktestEngine",
    "total_cost",
    "cost_savings_pct",
    "peak_feeder_power",
    "peak_reduction_pct",
    "comfort_score",
    "unmet_energy_kwh",
    "feeder_overload_energy_kwh",
    "feeder_overload_hours",
]
