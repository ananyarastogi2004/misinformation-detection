from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.fact_check import router as fact_check_router

app = FastAPI(
    title="ClaimSense API",
    version="2.0",
    description=(
        "AI-powered misinformation detection. "
        "9-stage pipeline: sarcasm detection → claim classification → "
        "evidence retrieval → FAISS ranking → credibility scoring → "
        "NLI verification → agreement analysis → Gemini web-search → "
        "hybrid verdict fusion."
    ),
    contact={"name": "ClaimSense Team"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # restrict to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(fact_check_router, prefix="/fact-check", tags=["Fact Check"])


@app.get("/", tags=["Health"])
def home():
    return {
        "message": "ClaimSense API v2.0 running 🚀",
        "docs":    "/docs",
        "redoc":   "/redoc"
    }