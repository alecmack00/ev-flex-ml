"""
Pydantic v2 schemas for API requests, responses, session payloads, and solver configurations.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class HealthCheck(BaseModel):
    """Health check status response."""
    status: str = Field("ok", description="Service health status")
    version: str = Field("0.1.0", description="API package version")
    timestamp: str = Field(..., description="Current server timestamp")


class SessionInput(BaseModel):
    """Input payload for a single EV charging session."""
    session_id: str = Field("SESS_0001", description="Unique session identifier")
    charger_id: str = Field("CP_01", description="Charger point identifier")
    arrival_time: str = Field("2024-01-01T08:00:00", description="ISO arrival timestamp")
    departure_time: str = Field("2024-01-01T17:00:00", description="ISO departure timestamp")
    battery_capacity_kwh: float = Field(60.0, gt=0, description="EV battery pack size in kWh")
    initial_soc: float = Field(0.2, ge=0.0, le=1.0, description="State of Charge at arrival")
    target_soc: float = Field(0.9, ge=0.0, le=1.0, description="Target State of Charge")
    required_energy_kwh: float = Field(42.0, ge=0, description="Energy target in kWh")
    max_charger_power_kw: float = Field(11.0, gt=0, description="Maximum charger power limit in kW")


class FeederConfig(BaseModel):
    """Feeder transformer capacity configuration."""
    feeder_id: str = Field("NL-AMS-FEEDER-04", description="Feeder ID")
    max_capacity_kw: float = Field(150.0, gt=0, description="Feeder transformer capacity limit in kW")
    safety_margin: float = Field(0.90, gt=0.0, le=1.0, description="Operational safety headroom factor")


class PriceSignalStep(BaseModel):
    """Single time step price entry."""
    step: int = Field(..., ge=0, description="Time step index")
    price_eur_kwh: float = Field(..., description="Electricity price in EUR/kWh")


class MDNPredictionRequest(BaseModel):
    """Request payload for MDN probabilistic predictions."""
    sessions: List[SessionInput]


class MDNPredictionResponse(BaseModel):
    """MDN probabilistic distribution output per session."""
    session_id: str
    expected_duration_hours: float
    std_duration_hours: float
    expected_energy_kwh: float
    std_energy_kwh: float


class QuantilePredictionResponse(BaseModel):
    """Quantile forecasting output."""
    quantiles: List[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])
    forecast: Dict[str, List[float]]


class DispatchRequest(BaseModel):
    """Payload to request an optimized charging dispatch schedule."""
    sessions: List[SessionInput]
    price_signals: List[float] = Field(..., description="Hourly or 15-min price vector in EUR/kWh")
    feeder_config: Optional[FeederConfig] = Field(default_factory=FeederConfig)
    dt_hours: float = Field(0.25, gt=0, description="Time step duration in hours")
    horizon_steps: int = Field(96, gt=0, description="Planning horizon length in time steps")


class DispatchScheduleItem(BaseModel):
    """Schedule for a single session across time steps."""
    session_id: str
    charger_id: str
    power_kw: List[float]


class DispatchResponse(BaseModel):
    """Optimization response containing dispatch schedule and performance metrics."""
    status: str
    total_cost_eur: float
    peak_load_kw: float
    total_unmet_kwh: float
    feeder_capacity_kw: float
    schedule: List[DispatchScheduleItem]
