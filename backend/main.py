# MODIFIED
"""
FastAPI application entry point.

On startup:
  • Checks whether backend/models/rf_model.pkl exists.
  • If not, triggers ModelTrainer to train and save it automatically.
  • Logs either "Model trained and saved" or "Model loaded from cache".
"""

import os

from fastapi import FastAPI                          # type: ignore[import]
from fastapi.middleware.cors import CORSMiddleware   # type: ignore[import]

from routers import upload, analyze                  # type: ignore[import]

app = FastAPI(title="Resume AI Detector API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    """Auto-train the RF model on first launch if no cached model exists."""
    model_path = os.path.join(os.path.dirname(__file__), "models", "rf_model.pkl")
    if not os.path.exists(model_path):
        print("[startup] No cached model found — training now...")
        try:
            from services.model import ModelTrainer  # type: ignore[import]
            ModelTrainer().train_and_save()
            print("[startup] Model trained and saved successfully.")
        except Exception as e:
            print(f"[startup] Model training failed: {e}. Will retry on first request.")
    else:
        print("[startup] Model loaded from cache.")


@app.get("/")
def read_root():
    return {"message": "Resume AI Detector API is running"}


app.include_router(upload.router)
app.include_router(analyze.router)
