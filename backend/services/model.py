# MODIFIED
"""
ResumeModel — trains RandomForest + XGBoost on Dataset A (text files + Excel),
caches both to disk, and returns ensemble + per-model predictions.

Falls back to a rule-based scorer if both ML models fail.
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

def _r4(x: float) -> float:
    return float(f"{float(x):.4f}")

# ── Paths ──────────────────────────────────────────────────────────────────
_MODELS_DIR         = os.path.join(_BACKEND_ROOT, "models")
_RF_PATH            = os.path.join(_MODELS_DIR, "rf_model.pkl")
_XGB_PATH           = os.path.join(_MODELS_DIR, "xgb_model.json")   # native XGBoost format
_FEATURE_NAMES_PATH = os.path.join(_MODELS_DIR, "feature_names.json")

# ── Label encoding ─────────────────────────────────────────────────────────
_LABEL_MAP: Dict[int, str] = {
    0: "human_written",
    1: "ai_generated",
    2: "template_based",
}
_LABEL_INT: Dict[str, int] = {v: k for k, v in _LABEL_MAP.items()}

# ── Thresholds for human-readable reasons ──────────────────────────────────
_THRESHOLDS: Dict[str, Tuple[str, float, str]] = {
    "type_token_ratio":             ("low",      0.45, "Low lexical diversity (repetitive vocabulary)."),
    "sentence_length_std":          ("low",      5.0,  "Very uniform sentence lengths (robotic writing flow)."),
    "avg_sentence_length":          ("high",    22.0,  "Unusually long average sentence length."),
    "perplexity":                   ("low",     40.0,  "Low neural perplexity — text is highly predictable (AI signal)."),
    "passive_voice_ratio":          ("high",    0.08,  "High passive voice usage detected."),
    "adjective_density":            ("high",    0.10,  "High adjective density (over-descriptive AI writing)."),
    "ai_phrase_match_count":        ("high",    2.0,   "Multiple AI signature phrases matched."),
    "template_pattern_match_count": ("high",    1.0,   "Template placeholder patterns detected."),
    "first_person_count":           ("low_zero", 0.0,  "No first-person pronouns (impersonal AI style)."),
    "informal_word_count":          ("low_zero", 0.0,  "No informal contractions found (overly formal AI text)."),
    "readability_flesch":           ("low",     40.0,  "Low Flesch readability score (unnaturally complex sentences)."),
}


# ────────────────────────────────────────────────────────────────────────────
# ModelTrainer
# ────────────────────────────────────────────────────────────────────────────

class ModelTrainer:
    """
    Trains RandomForestClassifier + XGBClassifier on Dataset A.
    Saves both models and shared feature_names to disk.
    """

    def train_and_save(self) -> Tuple[Any, Any, List[str]]:
        """Returns (rf_clf, xgb_clf, feature_names)."""
        os.makedirs(_MODELS_DIR, exist_ok=True)

        print("[ModelTrainer] Loading training data...")
        df = self._load_data()

        if df.empty or len(df) < 3:
            print("[ModelTrainer] Insufficient data — using synthetic fallback corpus.")
            df = self._synthetic_fallback()

        print(f"[ModelTrainer] Training on {len(df)} resumes "
              f"({df['label'].value_counts().to_dict()})")

        # Lazy WeakSupervision
        _ws_instance: Any = None
        try:
            from services.weak_supervision import WeakSupervision  # type: ignore[import]
            _ws_instance = WeakSupervision()
        except Exception as e:
            print(f"[ModelTrainer] WeakSupervision unavailable: {e}")

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

            feats = FeatureExtractor.extract_features(text, skip_perplexity=True, skip_spacy=True)

            if _ws_instance is not None:
                try:
                    h = _ws_instance.apply_heuristics(text)
                    feats["ai_phrase_match_count"]        = float(len(h.get("matched_ai_phrases", [])))
                    feats["template_pattern_match_count"] = float(len(h.get("matched_template_patterns", [])))
                except Exception:
                    pass

            rows.append(feats)
            labels.append(_LABEL_INT.get(label, 0))

        feature_names: List[str] = list(rows[0].keys()) if rows else []
        X = pd.DataFrame(rows, columns=feature_names).fillna(0.0).values
        y = np.array(labels)

        # ── Random Forest ─────────────────────────────────────────────
        from sklearn.ensemble import RandomForestClassifier  # type: ignore[import]

        rf = RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        )
        rf.fit(X, y)
        rf_train_acc = float((rf.predict(X) == y).mean())
        print(f"[ModelTrainer] RF  Train Accuracy: {rf_train_acc:.3f} (n={len(y)})")

        joblib.dump(rf, _RF_PATH)

        # ── XGBoost ───────────────────────────────────────────────────
        xgb_clf: Any = None
        try:
            from xgboost import XGBClassifier  # type: ignore[import]

            xgb_clf = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                eval_metric="mlogloss",
                random_state=42,
                verbosity=0,
                nthread=1,        # single-threaded — avoids OpenMP segfault on macOS
                tree_method="hist",
            )
            xgb_clf.fit(X, y)
            xgb_train_acc = float((xgb_clf.predict(X) == y).mean())
            print(f"[ModelTrainer] XGB Train Accuracy: {xgb_train_acc:.3f} (n={len(y)})")

            xgb_clf.save_model(_XGB_PATH)   # native JSON — no libomp at load time
            print(f"[ModelTrainer] XGBoost saved → {_XGB_PATH}")
        except Exception as xgb_err:
            print(f"[ModelTrainer] XGBoost training failed: {xgb_err}. Will use RF only.")

        with open(_FEATURE_NAMES_PATH, "w") as f:
            json.dump(feature_names, f)

        print(f"[ModelTrainer] RandomForest saved → {_RF_PATH}")
        return rf, xgb_clf, feature_names

    # ── Data loading ───────────────────────────────────────────────────────

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
            "Objective: To obtain a challenging position in a reputed organization. References available upon request.",
        ]
        records = (
            [{"resume_text": t, "label": "human_written",  "resume_id": f"SYN_H_{i}", "confidence_score": 0.9} for i, t in enumerate(human)] +
            [{"resume_text": t, "label": "ai_generated",   "resume_id": f"SYN_A_{i}", "confidence_score": 0.9} for i, t in enumerate(ai)]   +
            [{"resume_text": t, "label": "template_based", "resume_id": f"SYN_T_{i}", "confidence_score": 0.9} for i, t in enumerate(template)]
        )
        return pd.DataFrame(records)


# ────────────────────────────────────────────────────────────────────────────
# ResumeModel
# ────────────────────────────────────────────────────────────────────────────

class ResumeModel:
    """
    Loads cached RF + XGBoost models (trains if absent).
    predict() returns ensemble + per-model breakdowns.
    """

    def __init__(self) -> None:
        self._rf:  Any = None
        self._xgb: Any = None
        self._feature_names: List[str] = []
        self._load_or_train()

    # ── Initialization ─────────────────────────────────────────────────────

    def _load_or_train(self) -> None:
        loaded_rf  = False
        loaded_xgb = False

        if os.path.exists(_RF_PATH) and os.path.exists(_FEATURE_NAMES_PATH):
            try:
                self._rf = joblib.load(_RF_PATH)
                with open(_FEATURE_NAMES_PATH) as f:
                    self._feature_names = json.load(f)
                print("[ResumeModel] RandomForest loaded from cache.")
                loaded_rf = True
            except Exception as e:
                print(f"[ResumeModel] RF cache load failed: {e}")

        if os.path.exists(_XGB_PATH):
            try:
                from xgboost import XGBClassifier  # type: ignore[import]
                xgb_tmp = XGBClassifier()
                xgb_tmp.load_model(_XGB_PATH)      # native JSON — avoids libomp at import
                self._xgb = xgb_tmp
                print("[ResumeModel] XGBoost loaded from cache.")
                loaded_xgb = True
            except Exception as e:
                print(f"[ResumeModel] XGB cache load failed: {e}")

        if not loaded_rf:
            trainer = ModelTrainer()
            self._rf, self._xgb, self._feature_names = trainer.train_and_save()
        elif not loaded_xgb:
            # RF cached but XGBoost not — retrain both
            trainer = ModelTrainer()
            self._rf, self._xgb, self._feature_names = trainer.train_and_save()

    # ── Prediction ─────────────────────────────────────────────────────────

    def predict(self, text: str, heuristics: dict, skip_perplexity: bool = False) -> dict:
        try:
            feats = FeatureExtractor.extract_features(text, skip_perplexity=skip_perplexity, skip_spacy=False)
            feats["ai_phrase_match_count"]        = float(len(heuristics.get("matched_ai_phrases", [])))
            feats["template_pattern_match_count"] = float(len(heuristics.get("matched_template_patterns", [])))

            X = np.array([[feats.get(name, 0.0) for name in self._feature_names]])

            # ── Random Forest prediction ───────────────────────────────
            rf_result  = self._predict_single(self._rf,  X, "random_forest")

            # ── XGBoost prediction ─────────────────────────────────────
            xgb_result = self._predict_single(self._xgb, X, "xgboost")

            # ── Ensemble (soft vote: average probabilities) ────────────
            rf_proba  = rf_result.get("_proba",  np.array([0.34, 0.33, 0.33]))
            xgb_proba = xgb_result.get("_proba", np.array([0.34, 0.33, 0.33]))

            if self._xgb is not None:
                ensemble_proba = (rf_proba + xgb_proba) / 2.0
            else:
                ensemble_proba = rf_proba

            ens_idx    = int(np.argmax(ensemble_proba))
            ens_conf   = float(ensemble_proba[ens_idx])
            ens_label  = _LABEL_MAP.get(ens_idx, "human_written")
            is_ai      = ens_label in ("ai_generated", "template_based")

            # Top-5 importances from RF (most interpretable)
            top5 = self._top5_importances()

            reasons = self._generate_reasons(feats, heuristics, ens_label)

            return {
                "is_ai_generated":   is_ai,
                "label":             ens_label,
                "confidence":        _r4(ens_conf),
                "reasons":           reasons,
                "features":          feats,
                "feature_importances": top5,
                # Per-model breakdown
                "model_results": {
                    "random_forest": {
                        "label":      rf_result.get("label", "?"),
                        "confidence": _r4(rf_result.get("confidence", 0.0)),
                        "probabilities": {
                            _LABEL_MAP[i]: _r4(float(rf_proba[i]))
                            for i in range(len(rf_proba))
                        },
                    },
                    "xgboost": {
                        "label":      xgb_result.get("label", "unavailable"),
                        "confidence": _r4(xgb_result.get("confidence", 0.0)),
                        "probabilities": (
                            {
                                _LABEL_MAP[i]: _r4(float(xgb_proba[i]))
                                for i in range(len(xgb_proba))
                            }
                            if self._xgb is not None else {}
                        ),
                        "available": self._xgb is not None,
                    },
                    "ensemble": {
                        "label":      ens_label,
                        "confidence": _r4(ens_conf),
                        "method":     "soft_vote" if self._xgb is not None else "rf_only",
                        "probabilities": {
                            _LABEL_MAP[i]: _r4(float(ensemble_proba[i]))
                            for i in range(len(ensemble_proba))
                        },
                    },
                },
            }

        except Exception as e:
            print(f"[ResumeModel] Prediction error: {e}")
            return self._rule_based_fallback(text, heuristics)

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _predict_single(clf: Any, X: Any, name: str) -> dict:
        if clf is None:
            return {"label": "unavailable", "confidence": 0.0, "_proba": np.array([0.34, 0.33, 0.33])}
        try:
            proba    = clf.predict_proba(X)[0]
            pred_idx = int(np.argmax(proba))
            return {
                "label":      _LABEL_MAP.get(pred_idx, "human_written"),
                "confidence": float(proba[pred_idx]),
                "_proba":     proba,
            }
        except Exception as e:
            print(f"[ResumeModel] {name} predict_single error: {e}")
            return {"label": "error", "confidence": 0.0, "_proba": np.array([0.34, 0.33, 0.33])}

    def _top5_importances(self) -> Dict[str, float]:
        if self._rf is None:
            return {}
        importances = dict(zip(self._feature_names, self._rf.feature_importances_))
        sorted_imp  = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        top5: Dict[str, float] = {}
        for k, v in sorted_imp:
            if len(top5) >= 5:
                break
            top5[k] = _r4(v)
        return top5

    @staticmethod
    def _generate_reasons(feats: Dict[str, float], heuristics: dict, label: str) -> List[str]:
        reasons: List[str] = []
        for feat_name, (direction, threshold, msg) in _THRESHOLDS.items():
            val = feats.get(feat_name)
            if val is None:
                continue
            if direction == "low"      and val < threshold:
                reasons.append(msg)
            elif direction == "high"   and val > threshold:
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
            reasons.append(f"High perplexity ({int(ppl)}): unpredictable text — human signal.")

        if not reasons:
            reasons.append(f"Overall linguistic profile classified as: {label.replace('_', ' ')}.")

        return reasons

    @staticmethod
    def _rule_based_fallback(text: str, heuristics: dict) -> dict:
        score   = 0.0
        reasons: List[str] = []
        feats: Dict[str, float] = {}
        try:
            feats = FeatureExtractor.extract_features(text)
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
            reasons.append("Low lexical diversity.")
        ppl = feats.get("perplexity", 100.0)
        if 0 < ppl < 40:
            score += 0.3
            reasons.append(f"Low perplexity ({int(ppl)}): AI signal.")

        confidence = min(max(score, 0.0), 1.0)
        label      = "ai_generated" if confidence > 0.5 else "human_written"
        return {
            "is_ai_generated":   confidence > 0.5,
            "label":             label,
            "confidence":        _r4(confidence),
            "reasons":           reasons,
            "features":          feats,
            "feature_importances": {},
            "model_results": {
                "random_forest": {"label": label, "confidence": _r4(confidence), "probabilities": {}},
                "xgboost":       {"label": "unavailable", "confidence": 0.0, "probabilities": {}, "available": False},
                "ensemble":      {"label": label, "confidence": _r4(confidence), "method": "rule_based", "probabilities": {}},
            },
        }
