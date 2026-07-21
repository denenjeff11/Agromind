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
    """
    Returns the curated treatment info for a given disease key.
    Raises KeyError if the disease isn't in our data yet.
    """
    if disease_key not in TREATMENTS:
        raise KeyError(f"No treatment data for '{disease_key}' yet")
    return TREATMENTS[disease_key]
