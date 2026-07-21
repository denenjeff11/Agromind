from fastapi import FastAPI
from app.api.v1 import disease

app = FastAPI(title="Agromind")

app.include_router(disease.router, prefix="/api/v1", tags=["disease"])


@app.get("/health")
def health():
    return {"status": "ok"}
