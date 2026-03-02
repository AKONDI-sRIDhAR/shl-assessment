# api.py -- FastAPI backend
# Run: uvicorn api:app --host 0.0.0.0 --port 8080

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI(title="SHL Assessment Recommendation API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class RecReq(BaseModel):
    query: str
    top_k: Optional[int] = Field(8, ge=1, le=20)
    balance: Optional[bool] = True

class RecItem(BaseModel):
    rank: int
    name: str
    url: str
    score: float
    test_types: List[str]
    reason: str

class RecResp(BaseModel):
    query: str
    top_k: int
    balance: bool
    results: List[RecItem]


@app.get("/health")
def get_health():
    # lightweight check -- no model loading
    from utils import health
    return health()


@app.post("/recommend", response_model=RecResp)
def post_recommend(req: RecReq):
    # heavy imports happen lazily inside recommend() on first call
    from utils import recommend
    recs = recommend(req.query, top_k=req.top_k, balance=req.balance)
    items = [
        RecItem(
            rank=i+1,
            name=r.get("name",""),
            url=r.get("url",""),
            score=round(r.get("score",0), 4),
            test_types=r.get("test_types","").split(),
            reason=r.get("reason",""),
        )
        for i, r in enumerate(recs)
    ]
    return RecResp(query=req.query, top_k=req.top_k,
                   balance=req.balance, results=items)
