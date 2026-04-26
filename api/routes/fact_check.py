from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from services.pipeline import fact_check_pipeline

router = APIRouter()


class FactCheckRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Claim text to fact-check (10–1000 characters).",
        examples=["India is the most populous country in the world."]
    )


class FactCheckResponse(BaseModel):
    input:       str
    verdict:     str
    confidence:  float
    explanation: Optional[str] = None

    model_config = {"extra": "allow"}


@router.post("/", response_model=FactCheckResponse)
def fact_check(
    request: FactCheckRequest,
    disable: Optional[str] = Query(
        default=None,
        description="Ablation: disable one component. Values: gemini | nli | agreement"
    )
):
    """
    Fact-check endpoint with optional ablation support.

    Normal use:  POST /fact-check/
    Ablation:    POST /fact-check/?disable=gemini
                 POST /fact-check/?disable=nli
                 POST /fact-check/?disable=agreement
    """
    valid_disable = {None, "gemini", "nli", "agreement"}
    if disable not in valid_disable:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid disable value '{disable}'. Must be one of: gemini, nli, agreement"
        )

    try:
        result = fact_check_pipeline.run(
            request.text.strip(),
            disable=disable   # passed to pipeline
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
