"""
API endpoints for disease diagnosis and follow-up conversation.
"""
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel

from domains.disease_diagnosis.diagnose import diagnose_leaf
from domains.disease_diagnosis.treatment import get_treatment
from domains.disease_diagnosis.conversation import explain_diagnosis, continue_conversation

router = APIRouter()

UPLOAD_DIR = Path("data/raw/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/diagnose")
async def diagnose_plant(file: UploadFile = File(...)):
    image_path = UPLOAD_DIR / file.filename
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    diagnosis = diagnose_leaf(str(image_path))
    treatment = get_treatment(diagnosis["disease_key"])
    explanation = explain_diagnosis(diagnosis, treatment)

    return {
        "diagnosis": diagnosis,
        "treatment": treatment,
        "message": explanation,
    }


class FollowUpRequest(BaseModel):
    disease_key: str
    confidence: float
    is_healthy: bool
    chat_history: list = []
    question: str


@router.post("/diagnose/followup")
async def diagnose_followup(payload: FollowUpRequest):
    diagnosis = {
        "disease_key": payload.disease_key,
        "confidence": payload.confidence,
        "is_healthy": payload.is_healthy,
        "crop": "maize",
    }
    treatment = get_treatment(payload.disease_key)

    answer = continue_conversation(
        diagnosis, treatment, payload.chat_history, payload.question
    )

    return {
        "answer": answer,
    }
