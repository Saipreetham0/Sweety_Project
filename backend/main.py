# MODIFIED
"""
FastAPI application entry point.

On startup:
  • Checks whether backend/models/rf_model.pkl exists.
  • If not, triggers ModelTrainer to train and save it automatically.
  • Logs either "Model trained and saved" or "Model loaded from cache".
"""

import os
import subprocess
import sys

# Allow PyTorch and XGBoost to each load their own libomp without crashing.
# Both ship their own libomp.dylib on macOS; this env var tells OpenMP to
# tolerate duplicate library loads instead of segfaulting.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Import XGBoost BEFORE torch so its libomp loads first.
# If XGBoost isn't installed or fails, the server still runs (RF-only mode).
try:
    __import__("xgboost")  # pre-load XGBoost libomp before torch loads its own
except Exception:
    pass

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

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_RF_PATH     = os.path.join(_BACKEND_DIR, "models", "rf_model.pkl")
_XGB_PATH    = os.path.join(_BACKEND_DIR, "models", "xgb_model.json")


@app.on_event("startup")
async def startup_event() -> None:
    """
    Auto-train RF + XGBoost on first launch.
    Runs training as an isolated subprocess so PyTorch/Spacy don't
    conflict with the FastAPI server process (avoids fork-segfault).
    """
    models_missing = not os.path.exists(_RF_PATH) or not os.path.exists(_XGB_PATH)
    if models_missing:
        print("[startup] Models not found — training in subprocess...")
        try:
            result = subprocess.run(
                [sys.executable, os.path.join(_BACKEND_DIR, "train_models.py")],
                cwd=_BACKEND_DIR,
                capture_output=True,
                text=True,
                timeout=300,
            )
            print(result.stdout)
            if result.returncode == 0:
                print("[startup] RF + XGBoost trained and saved successfully.")
            else:
                print(f"[startup] Training subprocess failed (rc={result.returncode}):")
                print(result.stderr[-1000:] if result.stderr else "")
        except Exception as e:
            print(f"[startup] Training failed: {e}")
    else:
        print("[startup] Models loaded from cache (RF + XGBoost).")


@app.get("/")
def read_root():
    return {"message": "Resume AI Detector API is running"}


app.include_router(upload.router)
app.include_router(analyze.router)
