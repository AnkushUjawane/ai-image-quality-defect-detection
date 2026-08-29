from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.sql import func

from .database import Base


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    content_type = Column(String, nullable=True)
    file_size = Column(Integer, nullable=True)
    image_path = Column(String, nullable=True)  # relative path to stored thumbnail/copy
    quality_score = Column(Integer, nullable=False)
    quality_label = Column(String, nullable=False)
    issues_json = Column(Text, nullable=False)       # JSON-encoded list of issues
    image_stats_json = Column(Text, nullable=False)  # JSON-encoded engineered feature stats
    created_at = Column(DateTime(timezone=True), server_default=func.now())