"""
Unit tests for FastAPI REST endpoints using TestClient.
"""

from fastapi.testclient import TestClient
import pytest

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data


def test_predict_mdn_endpoint():
    payload = {
        "sessions": [
            {
                "session_id": "SESS_001",
                "charger_id": "CP_01",
                "arrival_time": "2024-01-01T08:00:00",
                "departure_time": "2024-01-01T17:00:00",
                "battery_capacity_kwh": 60.0,
                "initial_soc": 0.2,
                "target_soc": 0.9,
                "required_energy_kwh": 42.0,
                "max_charger_power_kw": 11.0,
            }
        ]
    }
    response = client.post("/predict/mdn", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["session_id"] == "SESS_001"
    assert data[0]["expected_duration_hours"] > 0


def test_predict_quantile_endpoint():
    response = client.post("/predict/quantile")
    assert response.status_code == 200
    data = response.json()
    assert "forecast" in data
    assert "q_0.5" in data["forecast"]
    assert len(data["forecast"]["q_0.5"]) == 24


def test_schedule_milp_endpoint():
    payload = {
        "sessions": [
            {
                "session_id": "SESS_001",
                "charger_id": "CP_01",
                "arrival_time": "2024-01-01T08:00:00",
                "departure_time": "2024-01-01T12:00:00",
                "battery_capacity_kwh": 60.0,
                "initial_soc": 0.2,
                "target_soc": 0.9,
                "required_energy_kwh": 30.0,
                "max_charger_power_kw": 11.0,
            }
        ],
        "price_signals": [0.20, 0.15, 0.10, 0.08, 0.12, 0.25, 0.30, 0.22, 0.18, 0.16, 0.15, 0.14, 0.12, 0.10, 0.09, 0.08],
        "dt_hours": 0.25,
        "horizon_steps": 16,
    }
    response = client.post("/schedule/milp", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["OPTIMAL", "optimal"]
    assert len(data["schedule"]) == 1
    assert len(data["schedule"][0]["power_kw"]) == 16


def test_schedule_mpc_endpoint():
    payload = {
        "sessions": [
            {
                "session_id": "SESS_001",
                "charger_id": "CP_01",
                "arrival_time": "2024-01-01T08:00:00",
                "departure_time": "2024-01-01T12:00:00",
                "battery_capacity_kwh": 60.0,
                "initial_soc": 0.2,
                "target_soc": 0.9,
                "required_energy_kwh": 30.0,
                "max_charger_power_kw": 11.0,
            }
        ],
        "price_signals": [0.20, 0.15, 0.10, 0.08, 0.12, 0.25, 0.30, 0.22, 0.18, 0.16, 0.15, 0.14, 0.12, 0.10, 0.09, 0.08],
        "dt_hours": 0.25,
        "horizon_steps": 16,
    }
    response = client.post("/schedule/mpc", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OPTIMAL"
    assert len(data["schedule"]) == 1
    assert len(data["schedule"][0]["power_kw"]) == 16
