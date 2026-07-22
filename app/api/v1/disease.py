"""
API endpoint for disease diagnosis.
"""
from fastapi import APIRouter, UploadFile, File
import shutil
from pathlib import Path

from domains.disease_diagnosis.diagnose import diagnose_leaf as diagnose
from domains.disease_diagnosis.treatment import get_treatment
from domains.disease_diagnosis.conversation import explain_diagnosis

router = APIRouter()

UPLOAD_DIR = Path("data/raw/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/diagnose")
async def diagnose_plant(file: UploadFile = File(...)):
    image_path = UPLOAD_DIR / file.filename
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    diagnosis = diagnose(str(image_path))
    
    # Safely extract the disease key regardless of internal dictionary structure
    disease_key = (
        diagnosis.get("disease_key") 
        or diagnosis.get("disease") 
        or diagnosis.get("label") 
        or "Healthy"
    )

    treatment = get_treatment(disease_key)
    explanation = explain_diagnosis(diagnosis, treatment)

    return {
        "diagnosis": diagnosis,
        "treatment": treatment,
        "message": explanation,
    }
