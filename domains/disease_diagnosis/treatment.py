"""
Loads the curated treatment data and looks up recommendations by disease key.
This is the SAFE, controlled data source - the LLM explains this data,
it does not invent chemical recommendations on its own.
"""
import json
from pathlib import Path

TREATMENTS_PATH = Path(__file__).resolve().parents[2] / "data" / "external" / "treatments.json"

with open(TREATMENTS_PATH, "r") as f:
    TREATMENTS = json.load(f)


def get_treatment(disease_key: str) -> dict:
    clean_key = disease_key.strip()

    # 1. Check if the model predicted a healthy plant
    if "healthy" in clean_key.lower():
        return {
            "disease": clean_key,
            "status": "Healthy",
            "chemical_treatment": "None required.",
            "organic_treatment": "None required.",
            "advice": "Your plant looks healthy! Continue standard watering, fertilization, and monitoring practices."
        }

    # 2. Return matching treatment if present in your treatment dictionary/JSON
    if clean_key in TREATMENTS:
        return TREATMENTS[clean_key]

    # 3. Safe fallback for infected plants whose specific treatment isn't mapped yet
    return {
        "disease": clean_key,
        "status": "Infected",
        "chemical_treatment": "Consult a local agricultural extension officer for target fungicides.",
        "organic_treatment": "Remove and destroy infected leaves to prevent spore spread.",
        "advice": "Isolate affected crops and monitor surrounding plants for early symptoms."
    }
