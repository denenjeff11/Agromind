from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import disease

app = FastAPI(title="Agromind")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for local testing; restrict this later
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(disease.router, prefix="/api/v1", tags=["disease"])


@app.get("/health")
def health():
    return {"status": "ok"}
