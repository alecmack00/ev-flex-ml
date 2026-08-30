"""
Unit tests for evaluation metrics and counterfactual backtesting engine.
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from src.data.data_loader import SyntheticDataGenerator
from src.evaluation.backtest import BacktestEngine
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


def test_metrics_calculations():
    # Feeder power and costs
    power_schedule = np.array([
        [10.0, 10.0, 0.0, 0.0],
        [0.0, 20.0, 20.0, 0.0],
    ])  # Aggregate loads: [10, 30, 20, 0] kW
    prices = np.array([0.10, 0.20, 0.15, 0.10])  # EUR/kWh
    dt_hours = 0.25

    cost = total_cost(power_schedule, prices, dt_hours)
    # Step costs: 10*0.10*0.25 (0.25) + 30*0.20*0.25 (1.50) + 20*0.15*0.25 (0.75) + 0 = 2.50 EUR
    assert np.isclose(cost, 2.50)

    peak = peak_feeder_power(power_schedule)
    assert peak == 30.0

    # Overload metrics with 25 kW capacity
    ov_kwh = feeder_overload_energy_kwh(power_schedule, feeder_capacity_kw=25.0, dt_hours=dt_hours)
    # Step 1: max(0, 30 - 25) = 5 kW * 0.25 h = 1.25 kWh
    assert np.isclose(ov_kwh, 1.25)

    ov_hrs = feeder_overload_hours(power_schedule, feeder_capacity_kw=25.0, dt_hours=dt_hours)
    # 1 step * 0.25 h = 0.25 h
    assert np.isclose(ov_hrs, 0.25)

    # Relative percentages
    savings = cost_savings_pct(smart_cost=80.0, baseline_cost=100.0)
    assert savings == 20.0

    peak_red = peak_reduction_pct(smart_peak=75.0, baseline_peak=100.0)
    assert peak_red == 25.0

    # Comfort score and unmet energy
    delivered = [15.0, 20.0]
    required = [20.0, 20.0]
    score = comfort_score(delivered, required)
    assert np.isclose(score, (35.0 / 40.0) * 100.0)

    unmet = unmet_energy_kwh(delivered, required)
    assert np.isclose(unmet, 5.0)


def test_backtest_engine_dataframe_input():
    gen = SyntheticDataGenerator(seed=42)
    df_sessions = gen.generate_sessions(num_sessions=10)
    prices = np.full(48, 0.15)
    prices[10:20] = 0.05

    engine = BacktestEngine(feeder_capacity_kw=100.0, dt_hours=0.25)
    df_summary, raw_results = engine.run_backtest_comparison(
        sessions=df_sessions,
        price_signal=prices,
        total_steps=48,
    )

    assert isinstance(df_summary, pd.DataFrame)
    assert len(df_summary) == 3
    assert "Overload Energy (kWh)" in df_summary.columns
    assert "Overload Duration (h)" in df_summary.columns
    assert "Unmanaged" in raw_results
    assert "TOU" in raw_results
    assert "Smart_MPC" in raw_results
    assert "overload_energy_kwh" in raw_results["Smart_MPC"]
    assert raw_results["Smart_MPC"]["overload_energy_kwh"] == 0.0


def test_backtest_engine_dict_timestamp_input():
    base_time = datetime(2024, 1, 1, 8, 0)
    sessions = [
        {
            "session_id": "SESS_01",
            "arrival_time": base_time.isoformat(),
            "departure_time": (base_time + timedelta(hours=4)).isoformat(),
            "required_energy_kwh": 20.0,
            "max_charger_power_kw": 11.0,
        },
        {
            "session_id": "SESS_02",
            "arrival_time": (base_time + timedelta(hours=1)).isoformat(),
            "departure_time": (base_time + timedelta(hours=6)).isoformat(),
            "required_energy_kwh": 30.0,
            "max_charger_power_kw": 11.0,
        },
    ]

    prices = np.full(32, 0.18)
    engine = BacktestEngine(feeder_capacity_kw=150.0, dt_hours=0.25)
    df_summary, raw_results = engine.run_backtest_comparison(
        sessions=sessions,
        price_signal=prices,
        total_steps=32,
    )

    assert len(df_summary) == 3
    assert raw_results["Unmanaged"]["total_cost_eur"] > 0.0
