# MODIFIED
"""
WeakSupervision — loads AI phrases and template patterns from Dataset B
(via load_datasets.py) instead of hardcoded lists.

Backward-compatible: keeps uses_ai_phrases, has_template_structure, perfect_grammar.
New keys: ai_phrase_score, matched_ai_phrases, template_score,
          matched_template_patterns, weak_label, weak_confidence.
"""

from __future__ import annotations

import os
import sys
from typing import List, Dict, TypedDict


class _PhraseEntry(TypedDict):
    phrase: str
    weight: float

# Ensure backend root is on path when this module is imported directly
_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

try:
    from data.load_datasets import load_ai_keywords, load_template_patterns
except ImportError:
    # Absolute fallback so the module never fails to load
    def load_ai_keywords() -> List[Dict]:  # type: ignore[misc]
        return []
    def load_template_patterns() -> List[Dict]:  # type: ignore[misc]
        return []

# Strength → float weight mapping
_STRENGTH_WEIGHTS: Dict[str, float] = {
    "very high": 1.0,
    "high":      0.75,
    "medium":    0.5,
    "low":       0.25,
}


class WeakSupervision:
    def __init__(self):
        ai_kw   = load_ai_keywords()
        tmpl_kw = load_template_patterns()

        self._ai_phrases: List[_PhraseEntry] = [
            _PhraseEntry(
                phrase=str(item.get("phrase", "")).lower().strip(),
                weight=float(_STRENGTH_WEIGHTS.get(
                    str(item.get("signal_strength", "medium")).lower().strip(), 0.5
                )),
            )
            for item in ai_kw
            if item.get("phrase")
        ]

        self._template_patterns: List[_PhraseEntry] = [
            _PhraseEntry(
                phrase=str(item.get("phrase", "")).lower().strip(),
                weight=float(_STRENGTH_WEIGHTS.get(
                    str(item.get("signal_strength", "medium")).lower().strip(), 0.5
                )),
            )
            for item in tmpl_kw
            if item.get("phrase")
        ]

        print(
            f"[WeakSupervision] Loaded {len(self._ai_phrases)} AI phrases "
            f"and {len(self._template_patterns)} template patterns."
        )

    # ------------------------------------------------------------------ #

    def apply_heuristics(self, text: str) -> dict:
        text_lower = text.lower()

        # ── AI phrase scoring ──────────────────────────────────────────
        matched_ai: List[str] = []
        ai_weights: List[float] = []
        for entry in self._ai_phrases:
            if entry["phrase"] and entry["phrase"] in text_lower:
                matched_ai.append(entry["phrase"])
                ai_weights.append(entry["weight"])

        ai_raw_score: float = sum(ai_weights)
        ai_denominator: float = max(len(self._ai_phrases) * 0.5, 1.0)
        ai_score: float = min(ai_raw_score / ai_denominator, 1.0)
        uses_ai_phrases = len(matched_ai) > 0

        # ── Template pattern scoring ───────────────────────────────────
        matched_tmpl: List[str] = []
        tmpl_weights: List[float] = []
        for entry in self._template_patterns:
            if entry["phrase"] and entry["phrase"] in text_lower:
                matched_tmpl.append(entry["phrase"])
                tmpl_weights.append(entry["weight"])

        tmpl_raw_score: float = sum(tmpl_weights)
        tmpl_denominator: float = max(len(self._template_patterns) * 0.5, 1.0)
        tmpl_score: float = min(tmpl_raw_score / tmpl_denominator, 1.0)

        # ── Structural template check (legacy) ─────────────────────────
        has_template_structure = self._check_structure(text_lower) or len(matched_tmpl) > 0

        # ── Weak label decision ────────────────────────────────────────
        if ai_score > 0.5:
            weak_label      = "ai_generated"
            weak_confidence = ai_score
        elif tmpl_score > 0.5:
            weak_label      = "template_based"
            weak_confidence = tmpl_score
        else:
            weak_label      = "uncertain"
            weak_confidence = max(ai_score, tmpl_score)

        return {
            # ── New keys ───────────────────────────────────────────────
            "uses_ai_phrases":            uses_ai_phrases,
            "ai_phrase_score":            round(ai_score, 4),
            "matched_ai_phrases":         matched_ai,
            "has_template_structure":     has_template_structure,
            "template_score":             round(tmpl_score, 4),
            "matched_template_patterns":  matched_tmpl,
            "weak_label":                 weak_label,
            "weak_confidence":            round(weak_confidence, 4),
            # ── Backward-compat keys ───────────────────────────────────
            "perfect_grammar":            False,
        }

    # ------------------------------------------------------------------ #

    @staticmethod
    def _check_structure(text_lower: str) -> bool:
        """Legacy: True when all 4 canonical resume sections are present."""
        required_sections = [
            ["professional summary", "summary", "objective"],
            ["skills", "core competencies", "technical skills"],
            ["experience", "work history", "employment"],
            ["education", "academic background"],
        ]
        found = sum(
            1 for group in required_sections
            if any(header in text_lower for header in group)
        )
        return found == 4
