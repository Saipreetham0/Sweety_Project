# MODIFIED
"""
FeatureExtractor — computes all 25 stylometric, structural, neural, and
heuristic-count features aligned to Dataset B "Stylometric Feature Vocabulary".

Main entry: FeatureExtractor.extract_features(text) -> dict
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List

import numpy as np  # type: ignore[import]
import textstat     # type: ignore[import]

# ── Pyright-safe rounding helper ───────────────────────────────────────────
# round(x, n) hits a Pyright overload-resolution bug when packages are in
# the system Python (not the venv). _r4 gives one clean float→float signature.
def _r4(x: float) -> float:
    return float(f"{float(x):.4f}")

# ── Spacy ──────────────────────────────────────────────────────────────────
try:
    import spacy as _spacy  # type: ignore[import]
    _nlp: Any = _spacy.load("en_core_web_sm")
except Exception:
    _nlp = None

# ── DistilGPT2 ─────────────────────────────────────────────────────────────
# Declared as Any (not Optional[Any]) so attribute access after a truthiness
# guard doesn't produce Pyright false positives.
_tokenizer: Any = None
_model: Any = None
try:
    import torch  # type: ignore[import]
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # type: ignore[import]
    _tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
    _model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    _model.eval()
except Exception as _gpt_err:
    print(f"[features] GPT-2 not available: {_gpt_err}")

# ── NLTK stopwords (function-word ratio) ───────────────────────────────────
try:
    from nltk.corpus import stopwords as _sw  # type: ignore[import]
    _STOPWORDS: set = set(_sw.words("english"))
except Exception:
    _STOPWORDS = set()

# ── Informal contractions & first-person pronouns ──────────────────────────
_CONTRACTIONS: set = {
    "don't", "doesn't", "can't", "won't", "i'm", "i've", "i'll",
    "i'd", "we're", "we've", "we'll", "they're", "they've",
    "you're", "you've", "isn't", "aren't", "wasn't", "weren't",
    "haven't", "hasn't", "hadn't", "shouldn't", "wouldn't", "couldn't",
    "it's", "that's", "there's",
}
_FIRST_PERSON: set = {"i", "me", "my", "myself", "mine"}


# ────────────────────────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Extracts a flat numeric feature dict from plain-text resume content.
    All features default to 0.0 on empty / error input — never raises.
    """

    @staticmethod
    def extract_features(text: str) -> Dict[str, float]:
        if not text or not text.strip():
            return FeatureExtractor._empty_features()

        feats: Dict[str, float] = {}

        feats.update(FeatureExtractor._stylometric(text))
        feats.update(FeatureExtractor._structural(text))

        feats["readability_flesch"] = float(textstat.flesch_reading_ease(text))
        feats["readability_fog"]    = float(textstat.gunning_fog(text))
        feats["reading_time"]       = float(textstat.reading_time(text))
        feats["perplexity"]         = FeatureExtractor._calculate_perplexity(text)

        feats.update(FeatureExtractor._informal_signals(text))

        # Heuristic counts — injected by model.py after WeakSupervision runs
        feats["ai_phrase_match_count"]        = 0.0
        feats["template_pattern_match_count"] = 0.0

        return feats

    # ──────────────────────────────────────────────────────────────────── #
    # Stylometric features
    # ──────────────────────────────────────────────────────────────────── #

    @staticmethod
    def _stylometric(text: str) -> Dict[str, float]:
        feats: Dict[str, float] = {}

        tokens_raw   = text.split()
        tokens_clean = [t.lower().strip(".,;:!?\"'()[]{}") for t in tokens_raw]
        tokens_clean = [t for t in tokens_clean if t]
        n_tokens = len(tokens_clean)

        if n_tokens == 0:
            return FeatureExtractor._zero_stylometric()

        # Type-token ratio
        ttr: float = len(set(tokens_clean)) / n_tokens
        feats["type_token_ratio"] = _r4(ttr)
        feats["lexical_richness"] = _r4(ttr)   # alias

        # Average word length
        feats["avg_word_length"] = _r4(
            sum(len(t) for t in tokens_clean) / n_tokens
        )

        # Function-word ratio (NLTK stopwords as proxy)
        if _STOPWORDS:
            fw: int = sum(1 for t in tokens_clean if t in _STOPWORDS)
            feats["function_word_ratio"] = _r4(fw / n_tokens)
        else:
            feats["function_word_ratio"] = 0.0

        # Comma density
        feats["comma_density"] = _r4(text.count(",") / n_tokens)

        # Passive voice: auxiliary + past-participle pattern
        passive: int = len(re.findall(
            r"\b(was|were|been|is|are|be)\s+\w+ed\b", text, re.IGNORECASE
        ))
        feats["passive_voice_ratio"] = _r4(passive / n_tokens)

        # Sentence-level (Spacy or regex fallback)
        if _nlp:
            try:
                text_chunk: str = text[0:100000]  # type: ignore[index]
                doc = _nlp(text_chunk)
                sent_lens: List[int] = [len(list(s)) for s in doc.sents]

                feats["avg_sentence_length"] = _r4(
                    float(np.mean(sent_lens)) if sent_lens else 0.0
                )
                feats["sentence_length_std"] = _r4(
                    float(np.std(sent_lens)) if sent_lens else 0.0
                )

                pos_counts: Counter = Counter(tok.pos_ for tok in doc)
                total: int = len(doc) or 1
                feats["noun_density"]      = _r4(pos_counts.get("NOUN", 0) / total)
                feats["adjective_density"] = _r4(pos_counts.get("ADJ",  0) / total)
                feats["verb_density"]      = _r4(pos_counts.get("VERB", 0) / total)

            except Exception:
                feats.update(FeatureExtractor._zero_spacy())
        else:
            sents  = re.split(r"[.!?]+", text)
            slens  = [len(s.split()) for s in sents if s.strip()]
            feats["avg_sentence_length"] = _r4(float(np.mean(slens)) if slens else 0.0)
            feats["sentence_length_std"] = _r4(float(np.std(slens))  if slens else 0.0)
            feats.update(FeatureExtractor._zero_spacy())

        return feats

    # ──────────────────────────────────────────────────────────────────── #
    # Structural features
    # ──────────────────────────────────────────────────────────────────── #

    @staticmethod
    def _structural(text: str) -> Dict[str, float]:
        lines   = text.splitlines()
        n_lines = max(len(lines), 1)

        bullet_lines = [
            ln.strip() for ln in lines
            if re.match(r"^\s*[•\-\*]\s+", ln)
        ]
        bullet_count = len(bullet_lines)

        if bullet_lines:
            bw: List[int] = [len(ln.split()) for ln in bullet_lines]
            avg_bullet_wc  = _r4(float(np.mean(bw)))
            bullet_len_var = _r4(float(np.std(bw)))
        else:
            avg_bullet_wc  = 0.0
            bullet_len_var = 0.0

        section_count: int = sum(
            1 for ln in lines
            if ln.strip() and (
                ln.strip().isupper()
                or (ln.strip().istitle() and len(ln.strip().split()) <= 4)
            )
        )

        caps_lines: int = sum(
            1 for ln in lines if ln.strip() and ln.strip().isupper()
        )
        all_caps_ratio = _r4(caps_lines / n_lines)

        return {
            "bullet_count":           float(bullet_count),
            "avg_bullet_word_count":  avg_bullet_wc,
            "bullet_length_variance": bullet_len_var,
            "section_count":          float(section_count),
            "all_caps_ratio":         all_caps_ratio,
        }

    # ──────────────────────────────────────────────────────────────────── #
    # Informal / first-person signals
    # ──────────────────────────────────────────────────────────────────── #

    @staticmethod
    def _informal_signals(text: str) -> Dict[str, float]:
        tokens = [t.lower().strip(".,;:!?\"'()") for t in text.split()]
        return {
            "first_person_count":  float(sum(1 for t in tokens if t in _FIRST_PERSON)),
            "informal_word_count": float(sum(1 for t in tokens if t in _CONTRACTIONS)),
        }

    # ──────────────────────────────────────────────────────────────────── #
    # Perplexity — DistilGPT2 sliding-window (original implementation kept)
    # ──────────────────────────────────────────────────────────────────── #

    @staticmethod
    def _calculate_perplexity(text: str) -> float:
        if not _model or not _tokenizer or not text.strip():
            return 0.0
        try:
            import torch as _torch  # type: ignore[import]
            encodings  = _tokenizer(text, return_tensors="pt")
            max_length = _model.config.n_positions
            stride     = 512
            seq_len    = encodings.input_ids.size(1)

            nlls: list = []
            prev_end_loc = 0
            for begin_loc in range(0, seq_len, stride):
                end_loc    = min(begin_loc + max_length, seq_len)
                trg_len    = end_loc - prev_end_loc
                input_ids  = encodings.input_ids[:, begin_loc:end_loc]
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                with _torch.no_grad():
                    outputs = _model(input_ids, labels=target_ids)
                    nlls.append(outputs.loss)

                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break

            if not nlls:
                return 0.0
            return float(_torch.exp(_torch.stack(nlls).mean()))
        except Exception:
            return 0.0

    # ──────────────────────────────────────────────────────────────────── #
    # Zero-value helpers
    # ──────────────────────────────────────────────────────────────────── #

    @staticmethod
    def _zero_spacy() -> Dict[str, float]:
        return {"noun_density": 0.0, "adjective_density": 0.0, "verb_density": 0.0}

    @staticmethod
    def _zero_stylometric() -> Dict[str, float]:
        return {
            "type_token_ratio": 0.0, "lexical_richness": 0.0,
            "avg_word_length": 0.0,  "function_word_ratio": 0.0,
            "comma_density": 0.0,    "passive_voice_ratio": 0.0,
            "avg_sentence_length": 0.0, "sentence_length_std": 0.0,
            "noun_density": 0.0, "adjective_density": 0.0, "verb_density": 0.0,
        }

    @staticmethod
    def _empty_features() -> Dict[str, float]:
        return {
            "type_token_ratio": 0.0,     "lexical_richness": 0.0,
            "avg_word_length": 0.0,      "function_word_ratio": 0.0,
            "comma_density": 0.0,        "passive_voice_ratio": 0.0,
            "avg_sentence_length": 0.0,  "sentence_length_std": 0.0,
            "noun_density": 0.0,         "adjective_density": 0.0,
            "verb_density": 0.0,
            "bullet_count": 0.0,         "avg_bullet_word_count": 0.0,
            "bullet_length_variance": 0.0, "section_count": 0.0,
            "all_caps_ratio": 0.0,
            "readability_flesch": 0.0,   "readability_fog": 0.0,
            "reading_time": 0.0,         "perplexity": 0.0,
            "first_person_count": 0.0,   "informal_word_count": 0.0,
            "ai_phrase_match_count": 0.0, "template_pattern_match_count": 0.0,
        }
