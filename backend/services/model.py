# MODIFIED
"""
ResumeModel — trains a RandomForest classifier on Dataset A on first run,
then caches it to backend/models/rf_model.pkl for subsequent requests.

Falls back to a rule-based scorer if the RF model fails.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple

import joblib        # type: ignore[import]
import numpy as np   # type: ignore[import]
import pandas as pd  # type: ignore[import]

# ── Path setup ─────────────────────────────────────────────────────────────
_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from services.features import FeatureExtractor

# Pyright-safe rounding helper (same reason as features.py)
def _r4(x: float) -> float:
    return float(f"{float(x):.4f}")

# ── Paths ──────────────────────────────────────────────────────────────────
_MODELS_DIR        = os.path.join(_BACKEND_ROOT, "models")
_MODEL_PATH        = os.path.join(_MODELS_DIR, "rf_model.pkl")
_FEATURE_NAMES_PATH = os.path.join(_MODELS_DIR, "feature_names.json")

# ── Label encoding ─────────────────────────────────────────────────────────
_LABEL_MAP: Dict[int, str] = {
    0: "human_written",
    1: "ai_generated",
    2: "template_based",
}
_LABEL_INT: Dict[str, int] = {v: k for k, v in _LABEL_MAP.items()}

# ── Feature thresholds for generating human-readable reasons ───────────────
# Format: feature_name → (direction, threshold, message)
# direction: "low" = AI when value < threshold
#            "high" = AI when value > threshold
#            "low_zero" = AI when value == 0
_THRESHOLDS: Dict[str, Tuple[str, float, str]] = {
    "type_token_ratio":           ("low",      0.45, "Low lexical diversity (repetitive vocabulary)."),
    "lexical_richness":           ("low",      0.45, "Low lexical richness detected."),
    "sentence_length_std":        ("low",      5.0,  "Very uniform sentence lengths (robotic writing flow)."),
    "avg_sentence_length":        ("high",    22.0,  "Unusually long average sentence length."),
    "perplexity":                 ("low",     40.0,  "Low neural perplexity — text is highly predictable (AI signal)."),
    "passive_voice_ratio":        ("high",    0.08,  "High passive voice usage detected."),
    "adjective_density":          ("high",    0.10,  "High adjective density (over-descriptive AI writing)."),
    "ai_phrase_match_count":      ("high",    2.0,   "Multiple AI signature phrases matched."),
    "template_pattern_match_count":("high",   1.0,   "Template placeholder patterns detected."),
    "first_person_count":         ("low_zero", 0.0,  "No first-person pronouns (impersonal AI writing style)."),
    "informal_word_count":        ("low_zero", 0.0,  "No informal contractions found (overly formal AI text)."),
    "readability_flesch":         ("low",     40.0,  "Low Flesch readability score (unnaturally complex sentence structure)."),
}


# ────────────────────────────────────────────────────────────────────────────
# ModelTrainer
# ────────────────────────────────────────────────────────────────────────────

class ModelTrainer:
    """Trains a RandomForestClassifier on Dataset A and saves it to disk."""

    def train_and_save(self) -> Tuple[Any, List[str]]:
        os.makedirs(_MODELS_DIR, exist_ok=True)

        print("[ModelTrainer] Loading training data from Dataset A...")
        df = self._load_data()

        if df.empty or len(df) < 3:
            print("[ModelTrainer] Insufficient data — using synthetic fallback corpus.")
            df = self._synthetic_fallback()

        # Lazy-import WeakSupervision to avoid circular-import risk
        _ws_instance: Any = None
        try:
            from services.weak_supervision import WeakSupervision  # type: ignore[import]
            _ws_instance = WeakSupervision()
        except Exception as e:
            print(f"[ModelTrainer] WeakSupervision not available: {e}")
        ws = _ws_instance  # kept as Any; truthiness guard below is safe

        print(f"[ModelTrainer] Extracting features from {len(df)} resumes...")
        rows: List[Dict[str, float]] = []
        labels: List[int] = []

        for _, row in df.iterrows():
            text      = str(row.get("resume_text", ""))
            label_raw = str(row.get("label", "human_written")).strip().lower()

            if "human" in label_raw:
                label = "human_written"
            elif "template" in label_raw:
                label = "template_based"
            else:
                label = "ai_generated"

            feats = FeatureExtractor.extract_features(text)

            if ws is not None:
                try:
                    h = ws.apply_heuristics(text)
                    feats["ai_phrase_match_count"]        = float(len(h.get("matched_ai_phrases", [])))
                    feats["template_pattern_match_count"] = float(len(h.get("matched_template_patterns", [])))
                except Exception:
                    pass

            rows.append(feats)
            labels.append(_LABEL_INT.get(label, 0))

        feature_names: List[str] = list(rows[0].keys()) if rows else []
        X = pd.DataFrame(rows, columns=feature_names).fillna(0.0).values
        y = np.array(labels)

        from sklearn.ensemble import RandomForestClassifier  # type: ignore[import]
        from sklearn.model_selection import cross_val_score  # type: ignore[import]

        clf = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        )
        clf.fit(X, y)

        n_splits = min(3, len(X))
        if n_splits >= 2:
            scores = cross_val_score(clf, X, y, cv=n_splits, scoring="accuracy")
            print(f"[ModelTrainer] CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
        else:
            print(f"[ModelTrainer] Training accuracy: {clf.score(X, y):.3f}")

        joblib.dump(clf, _MODEL_PATH)
        with open(_FEATURE_NAMES_PATH, "w") as f:
            json.dump(feature_names, f)

        print(f"[ModelTrainer] Model saved → {_MODEL_PATH}")
        return clf, feature_names

    # ── Data loading ───────────────────────────────────────────────────

    @staticmethod
    def _load_data() -> pd.DataFrame:
        try:
            from data.load_datasets import load_training_data  # type: ignore[import]
            return load_training_data()
        except Exception as e:
            print(f"[ModelTrainer] Could not load Dataset A: {e}")
            return pd.DataFrame()

    @staticmethod
    def _synthetic_fallback() -> pd.DataFrame:
        """Minimal corpus for zero-shot bootstrapping when Dataset A is absent."""
        human = [
            "I worked at Google for 3 years. I built the payments API using Java. I collaborated daily with my team.",
            "My experience at Microsoft taught me a lot. I can't imagine a better place to grow as an engineer.",
            "Graduated from MIT in 2019. I've always been passionate about building products people love.",
            "I led a small team of 4 engineers and we shipped the mobile app on time. My proudest achievement.",
        ]
        ai = [
            "Results-driven professional with a proven track record of spearheading cross-functional initiatives leveraging cutting-edge technologies.",
            "Dynamic and detail-oriented software engineer poised to deliver exceptional value in a fast-paced environment by leveraging the power of modern frameworks.",
            "Meticulously crafted resume showcasing an unwavering commitment to excellence, navigating the complexities of enterprise software development.",
            "Highly motivated self-starter with strong communication skills and a passion for innovative solutions in today's fast-paced landscape.",
        ]
        template = [
            "[Company Name] | [Position Title] | [Date Range] References available upon request. Seeking a challenging position in a reputed organization.",
            "I hereby declare that all information provided above is true to the best of my knowledge. [Your Name] [Date]",
            "Objective: To obtain a challenging position in a reputed organization. References available upon request. [Company Name]",
        ]
        records = (
            [{"resume_text": t, "label": "human_written",  "resume_id": f"SYN_H_{i}", "confidence_score": 0.9} for i, t in enumerate(human)] +
            [{"resume_text": t, "label": "ai_generated",   "resume_id": f"SYN_A_{i}", "confidence_score": 0.9} for i, t in enumerate(ai)] +
            [{"resume_text": t, "label": "template_based", "resume_id": f"SYN_T_{i}", "confidence_score": 0.9} for i, t in enumerate(template)]
        )
        return pd.DataFrame(records)


# ────────────────────────────────────────────────────────────────────────────
# ResumeModel
# ────────────────────────────────────────────────────────────────────────────

class ResumeModel:
    """
    Loads the cached RandomForest model or triggers training on first run.
    predict() returns a structured dict consumed by analyze.py.
    """

    def __init__(self) -> None:
        self._clf: Any = None
        self._feature_names: List[str] = []
        self._load_or_train()

    # ── Initialization ─────────────────────────────────────────────────

    def _load_or_train(self) -> None:
        if os.path.exists(_MODEL_PATH) and os.path.exists(_FEATURE_NAMES_PATH):
            try:
                self._clf = joblib.load(_MODEL_PATH)
                with open(_FEATURE_NAMES_PATH) as f:
                    self._feature_names = json.load(f)
                print("[ResumeModel] Model loaded from cache.")
                return
            except Exception as e:
                print(f"[ResumeModel] Cache load failed ({e}) — retraining...")

        trainer = ModelTrainer()
        self._clf, self._feature_names = trainer.train_and_save()

    # ── Prediction ─────────────────────────────────────────────────────

    def predict(self, text: str, heuristics: dict) -> dict:
        try:
            feats = FeatureExtractor.extract_features(text)

            # Inject heuristic counts into feature vector
            feats["ai_phrase_match_count"]        = float(len(heuristics.get("matched_ai_phrases", [])))
            feats["template_pattern_match_count"] = float(len(heuristics.get("matched_template_patterns", [])))

            # Align to training feature order
            X = np.array([[feats.get(name, 0.0) for name in self._feature_names]])

            proba    = self._clf.predict_proba(X)[0]
            pred_idx = int(np.argmax(proba))
            confidence = float(proba[pred_idx])
            label    = _LABEL_MAP.get(pred_idx, "human_written")
            is_ai    = label in ("ai_generated", "template_based")

            # Top-5 feature importances (avoid list[:n] Pyright slice bug)
            importances: Dict[str, float] = dict(
                zip(self._feature_names, self._clf.feature_importances_)
            )
            sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            top5: Dict[str, float] = {}
            for k, v in sorted_imp:
                if len(top5) >= 5:
                    break
                top5[k] = v

            reasons = self._generate_reasons(feats, heuristics, label)

            return {
                "is_ai_generated":   is_ai,
                "label":             label,
                "confidence":        _r4(confidence),
                "reasons":           reasons,
                "features":          feats,
                "feature_importances": {k: _r4(v) for k, v in top5.items()},
            }

        except Exception as e:
            print(f"[ResumeModel] Prediction error: {e}")
            return self._rule_based_fallback(text, heuristics)

    # ── Reason generation ──────────────────────────────────────────────

    @staticmethod
    def _generate_reasons(
        feats: Dict[str, float],
        heuristics: dict,
        label: str,
    ) -> List[str]:
        reasons: List[str] = []

        for feat_name, (direction, threshold, msg) in _THRESHOLDS.items():
            val = feats.get(feat_name)
            if val is None:
                continue
            if direction == "low" and val < threshold:
                reasons.append(msg)
            elif direction == "high" and val > threshold:
                reasons.append(msg)
            elif direction == "low_zero" and val == 0:
                reasons.append(msg)

        matched_ai = heuristics.get("matched_ai_phrases", [])
        if matched_ai:
            sample = '", "'.join(matched_ai[:3])
            reasons.append(f'Matched AI phrases: "{sample}".')

        matched_tmpl = heuristics.get("matched_template_patterns", [])
        if matched_tmpl:
            sample = '", "'.join(matched_tmpl[:3])
            reasons.append(f'Matched template patterns: "{sample}".')

        ppl = feats.get("perplexity", 0)
        if ppl and ppl > 80:
            reasons.append(
                f"High perplexity ({int(ppl)}): unpredictable text — human signal."
            )

        if not reasons:
            reasons.append(
                f"Overall linguistic profile classified as: {label.replace('_', ' ')}."
            )

        return reasons

    # ── Rule-based fallback ────────────────────────────────────────────

    @staticmethod
    def _rule_based_fallback(text: str, heuristics: dict) -> dict:
        """Used when RF prediction fails; mirrors the original scoring logic."""
        score  = 0.0
        reasons: List[str] = []
        feats: Dict[str, float] = {}

        try:
            feats = FeatureExtractor.extract_features(text)
            feats["ai_phrase_match_count"]        = float(len(heuristics.get("matched_ai_phrases", [])))
            feats["template_pattern_match_count"] = float(len(heuristics.get("matched_template_patterns", [])))
        except Exception:
            pass

        if heuristics.get("uses_ai_phrases"):
            score += 0.4
            reasons.append("Contains AI-generated phrases.")
        if heuristics.get("has_template_structure"):
            score += 0.2
            reasons.append("Follows a generic template structure.")
        if feats.get("type_token_ratio", 0.5) < 0.4:
            score += 0.2
            reasons.append("Low lexical diversity (repetitive vocabulary).")
        if feats.get("sentence_length_std", 10.0) < 5.0:
            score += 0.1
            reasons.append("Very uniform sentence lengths (robotic flow).")

        ppl = feats.get("perplexity", 100.0)
        if 0 < ppl < 40:
            score += 0.3
            reasons.append(f"Low perplexity ({int(ppl)}): AI signal.")
        elif ppl > 80:
            score -= 0.1
            reasons.append(f"High perplexity ({int(ppl)}): human signal.")

        confidence = min(max(score, 0.0), 1.0)
        label      = "ai_generated" if confidence > 0.5 else "human_written"

        return {
            "is_ai_generated":     confidence > 0.5,
            "label":               label,
            "confidence":          _r4(confidence),
            "reasons":             reasons,
            "features":            feats,
            "feature_importances": {},
        }
