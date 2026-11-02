"""
FastAPI application entry point.

Models are loaded (or trained, if backend/models/ is empty) when
routers.analyze is imported — see services/model.py:ResumeModel.
"""

import os

# Allow PyTorch and XGBoost to each load their own libomp without crashing.
# Both ship their own libomp.dylib on macOS; this env var tells OpenMP to
# tolerate duplicate library loads instead of segfaulting.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Import XGBoost BEFORE torch so its libomp loads first.
import xgboost  # noqa: F401

from fastapi import FastAPI                          # type: ignore[import]
from fastapi.middleware.cors import CORSMiddleware   # type: ignore[import]

from routers import upload, analyze                  # type: ignore[import]

app = FastAPI(title="Resume AI Detector API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Resume AI Detector API is running"}


app.include_router(upload.router)
app.include_router(analyze.router)
