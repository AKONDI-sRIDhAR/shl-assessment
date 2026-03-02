# api.py -- FastAPI backend
# Run: uvicorn api:app --host 0.0.0.0 --port 8000

import os
import threading
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI(title="SHL Assessment Recommendation API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# pre-load model in background so it's ready when /recommend is called
def _preload():
    from utils import get_model, get_index, get_meta
    get_meta()
    get_index()
    get_model()  # heaviest, load last

@app.on_event("startup")
def startup():
    threading.Thread(target=_preload, daemon=True).start()


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


@app.get("/")
def root():
    return {"message": "SHL Assessment Recommendation API",
            "usage": "POST /recommend with {query, top_k, balance}"}


@app.get("/health")
def get_health():
    from utils import health
    return health()


def _do_recommend(query, top_k=8, balance=True):
    from utils import recommend
    recs = recommend(query, top_k=top_k, balance=balance)
    items = [
        RecItem(rank=i+1, name=r.get("name",""), url=r.get("url",""),
                score=round(r.get("score",0), 4),
                test_types=r.get("test_types","").split(),
                reason=r.get("reason",""))
        for i, r in enumerate(recs)
    ]
    return RecResp(query=query, top_k=top_k, balance=balance, results=items)


@app.post("/recommend", response_model=RecResp)
def post_recommend(req: RecReq):
    return _do_recommend(req.query, req.top_k, req.balance)


@app.get("/recommend", response_model=RecResp)
def get_recommend(query: str, top_k: int = 8, balance: bool = True):
    return _do_recommend(query, top_k, balance)
