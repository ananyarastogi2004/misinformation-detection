from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.pipeline import fact_check_pipeline

router = APIRouter()


class FactCheckRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="The claim text to fact-check (10–1000 characters).",
        examples=["India is the most populous country in the world."]
    )


class FactCheckResponse(BaseModel):
    input:       str
    verdict:     str
    confidence:  float
    explanation: Optional[str] = None   # FIX: str | None requires Python 3.10+
                                        # Optional[str] works on Python 3.9 and below

    model_config = {"extra": "allow"}


@router.post("/", response_model=FactCheckResponse)
def fact_check(request: FactCheckRequest):
    """
    Main fact-check endpoint. Returns a 5-class verdict:
    TRUE / LIKELY TRUE / UNCERTAIN / LIKELY FALSE / FALSE
    """
    try:
        result = fact_check_pipeline.run(request.text.strip())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")