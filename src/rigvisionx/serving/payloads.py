"""Request/response payload definitions"""

from typing import Optional, List
from pydantic import BaseModel


class SensorData(BaseModel):
    """Sensor data payload."""
    sensor_id: str
    value: float
    unit: str
    timestamp: str


class PredictionRequest(BaseModel):
    """Request for predictions."""
    equipment_id: str
    sensor_data: List[SensorData]
    include_analysis: bool = False


class PredictionResponse(BaseModel):
    """Response with predictions."""
    equipment_id: str
    failure_risk: float
    rul_days: Optional[int]
    anomaly_score: float
    risk_level: str
    recommended_actions: List[str]
    timestamp: str


class ExplainPredictionRequest(BaseModel):
    """Request for prediction explanation."""
    equipment_id: str
    sensor_data: List[SensorData]
    model_type: str = "failure_risk"  # failure_risk, rul, anomaly
    top_features: int = 10


class FeatureContribution(BaseModel):
    """Feature contribution to prediction."""
    feature: str
    value: float
    shap_contribution: float
    impact: str  # "increases" or "decreases"


class ExplainPredictionResponse(BaseModel):
    """Response with prediction explanation."""
    equipment_id: str
    model_type: str
    prediction_value: float
    base_value: float
    feature_contributions: List[FeatureContribution]
    timestamp: str


class FeatureImportanceRequest(BaseModel):
    """Request for feature importance."""
    model_type: str = "failure_risk"
    top_n: int = 15


class FeatureImportance(BaseModel):
    """Feature importance entry."""
    feature: str
    importance: float


class FeatureImportanceResponse(BaseModel):
    """Response with feature importance."""
    model_type: str
    top_features: List[FeatureImportance]
