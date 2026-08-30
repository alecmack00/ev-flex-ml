"""
Unit tests for mathematical optimization solvers (MILPScheduler, FeederConstraintManager, MPCController).
"""

import numpy as np
import pytest

from src.optimization.constraints import FeederConstraintManager
from src.optimization.milp_scheduler import MILPScheduler
from src.optimization.mpc_controller import MPCController


def test_feeder_constraint_manager_validation():
    mgr = FeederConstraintManager(feeder_capacity_kw=100.0, safety_margin=0.9)

    # Schedule within feeder limits
    matrix = np.full((5, 10), 10.0)  # 5 * 10 kW = 50 kW total load (under 90 kW limit)
    sessions = [
        {"arr_step": 0, "dep_step": 10, "required_energy_kwh": 20.0, "max_charger_power_kw": 11.0}
        for _ in range(5)
    ]

    res = mgr.validate_schedule(matrix, sessions, dt_hours=0.25)
    assert res["max_feeder_load_kw"] == 50.0
    assert res["feeder_limit_kw"] == 90.0

    # Overloaded schedule
    matrix_overload = np.full((10, 10), 11.0)  # 10 * 11 kW = 110 kW total load (exceeds 90 kW limit)
    res_ov = mgr.validate_schedule(matrix_overload, sessions, dt_hours=0.25)
    assert res_ov["max_feeder_load_kw"] == 110.0
    assert not res_ov["is_valid"]
    assert res_ov["violation_count"] > 0


def test_milp_scheduler_solve():
    scheduler = MILPScheduler(feeder_capacity_kw=100.0, dt_hours=0.25)

    sessions = [
        {
            "session_id": "SESS_1",
            "arr_step": 0,
            "dep_step": 8,  # 2 hours plug window
            "required_energy_kwh": 10.0,
            "max_charger_power_kw": 11.0,
        },
        {
            "session_id": "SESS_2",
            "arr_step": 2,
            "dep_step": 10,
            "required_energy_kwh": 15.0,
            "max_charger_power_kw": 11.0,
        },
    ]

    prices = [0.20, 0.18, 0.12, 0.10, 0.08, 0.15, 0.25, 0.30, 0.22, 0.20]

    sol = scheduler.solve(sessions, prices, horizon_steps=10)

    assert sol["status"] in ["OPTIMAL", "optimal"]
    assert sol["power_matrix"].shape == (2, 10)
    assert sol["total_cost_eur"] > 0.0
    assert sol["peak_load_kw"] <= 100.0


def test_mpc_controller_simulation():
    mpc = MPCController(feeder_capacity_kw=120.0, horizon_steps=12, dt_hours=0.25)

    sessions = [
        {
            "session_id": "SESS_1",
            "arr_step": 0,
            "dep_step": 8,
            "initial_soc": 0.2,
            "battery_capacity_kwh": 50.0,
            "required_energy_kwh": 15.0,
            "max_charger_power_kw": 11.0,
        }
    ]

    prices = np.full(24, 0.15)
    prices[2:6] = 0.08  # Low price window

    res = mpc.run_simulation(sessions, prices, total_steps=12)

    assert res["dispatch_matrix"].shape == (1, 12)
    assert res["soc_history"].shape == (1, 13)
    assert res["total_cost_eur"] >= 0.0


def test_mpc_ml_fallback():
    from src.models.mdn_network import MixtureDensityNetwork
    mdn_model = MixtureDensityNetwork(input_dim=12)
    mpc = MPCController(feeder_capacity_kw=100.0, horizon_steps=8, ml_model=mdn_model)

    # Session missing dep_step and required_energy_kwh
    incomplete_session = {
        "session_id": "SESS_MISSING",
        "arr_step": 0,
        "battery_capacity_kwh": 60.0,
        "initial_soc": 0.1,
    }

    dep_step, req_energy = mpc.predict_session_params(incomplete_session, current_step=0)
    assert dep_step > 0
    assert req_energy > 0.0
