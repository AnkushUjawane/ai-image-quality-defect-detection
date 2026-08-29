from typing import List, Optional
from pydantic import BaseModel


class Issue(BaseModel):
    type: str
    severity: str
    confidence: float
    explanation: str


class AnalysisResult(BaseModel):
    id: int
    filename: str
    quality_score: int
    quality_label: str
    issues: List[Issue]
    image_stats: dict
    image_url: Optional[str] = None
    created_at: str

    class Config:
        from_attributes = True


class AnalysisListItem(BaseModel):
    id: int
    filename: str
    quality_score: int
    quality_label: str
    issue_count: int
    image_url: Optional[str] = None
    created_at: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str