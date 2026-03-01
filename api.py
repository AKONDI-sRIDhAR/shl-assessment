"""
api.py -- FastAPI backend for the SHL Assessment Recommendation Engine

Endpoints:
    GET  /health     -> {"status": "healthy"}
    POST /recommend  -> list of recommended assessments

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from utils import recommend, health_check

app = FastAPI(
    title="SHL Assessment Recommendation API",
    version="1.0",
    description="Recommend SHL assessments based on natural-language queries.",
)

# allow cross-origin requests from the Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- request / response schemas --

class RecommendRequest(BaseModel):
    query: str = Field(..., description="Natural-language job description or query")
    top_k: Optional[int] = Field(8, ge=1, le=20, description="Number of results")
    balance: Optional[bool] = Field(True, description="Balance K vs P test types")


class AssessmentResult(BaseModel):
    rank: int
    name: str
    url: str
    score: float
    test_types: List[str]
    reason: str


class RecommendResponse(BaseModel):
    query: str
    top_k: int
    balance: bool
    results: List[AssessmentResult]


class HealthResponse(BaseModel):
    status: str


# -- endpoints --

@app.get("/health", response_model=HealthResponse)
def get_health():
    info = health_check()
    return {"status": info["status"]}


@app.post("/recommend", response_model=RecommendResponse)
def post_recommend(req: RecommendRequest):
    recs = recommend(req.query, top_k=req.top_k, balance=req.balance)

    results = [
        AssessmentResult(
            rank=i + 1,
            name=r.get("name", ""),
            url=r.get("url", ""),
            score=round(r.get("score", 0), 4),
            test_types=r.get("test_types", "").split(),
            reason=r.get("reason", ""),
        )
        for i, r in enumerate(recs)
    ]

    return RecommendResponse(
        query=req.query,
        top_k=req.top_k,
        balance=req.balance,
        results=results,
    )
