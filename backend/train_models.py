"""
Standalone training script — run as an isolated subprocess so PyTorch/Spacy
don't conflict with the FastAPI server process.

Usage:
    python3 train_models.py
"""

import sys
import os

# Skip loading DistilGPT2 during batch training to avoid PyTorch/Spacy
# fork-segfault on macOS. Perplexity is computed per-request at inference time.
os.environ["SKIP_GPT2"]   = "1"
os.environ["SKIP_SPACY"]  = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.model import ModelTrainer

print("[train_models] Starting RF + XGBoost training...")
rf, xgb, feats = ModelTrainer().train_and_save()
print(f"[train_models] Done. RF={type(rf).__name__}, XGB={type(xgb).__name__ if xgb else 'None'}, features={len(feats)}")
