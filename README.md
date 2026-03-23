# AI Resume Detector

A research-grade system for detecting AI-generated resumes using a **Weakly Supervised Hybrid ML Pipeline**. Combines neural perplexity scoring, stylometric feature extraction, heuristic weak supervision (Dataset B), and a trained Random Forest classifier (Dataset A).

## Scientific Methodology

### 1. Neural Perplexity
Uses **DistilGPT2** (HuggingFace Transformers) to measure how statistically predictable the text is.
- Low perplexity (< 40): AI-generated text tends to be smooth and predictable.
- High perplexity (> 80): Human writing is more varied and "bursty".

### 2. Stylometric Analysis (25 Features)
Extracts a 25-dimensional feature vector per resume:
- **Lexical**: Type-Token Ratio, avg word length, function word ratio
- **Structural**: Bullet count, bullet length variance, all-caps ratio, section count
- **Syntactic**: Passive voice ratio, noun/adjective/verb density, POS burstiness
- **Readability**: Flesch Reading Ease, Gunning Fog Index, reading time
- **Heuristic counts**: AI phrase match count, template pattern match count

### 3. Weak Supervision (Dataset B)
`Dataset_B_Keyword_Heuristic_Signals.xlsx` provides:
- **50 AI Signature Phrases** with signal weights (Very High=1.0, High=0.75, Medium=0.5, Low=0.25) — e.g., *"delve into"*, *"meticulously crafted"*, *"proven track record"*
- **24 Template Pattern Keywords** for detecting rigid CV template structures

### 4. Random Forest Classifier (Dataset A)
`Dataset_A_Combined_Resume_Corpus.xlsx` (sheet: "Merged Dataset (Training)") trains a `RandomForestClassifier(n_estimators=100, class_weight='balanced')` producing **3-class output**:

| Label | Meaning |
|---|---|
| `human_written` | Resume authored by a human |
| `ai_generated` | Resume produced by a language model |
| `template_based` | Resume from a rigid template / AI-assisted tool |

The model is auto-trained on first startup and cached to `backend/models/rf_model.pkl`. If Dataset A is absent, a synthetic 11-record fallback corpus bootstraps training.

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full Mermaid diagram.

### Backend (Python / FastAPI)
- `services/extraction.py` — PDF/DOCX → raw text
- `services/preprocessing.py` — normalisation, stop-word removal
- `services/weak_supervision.py` — Dataset B heuristics
- `services/features.py` — 25-feature vector extraction
- `services/model.py` — RandomForest training (`ModelTrainer`) and inference (`ResumeModel`)
- `data/load_datasets.py` — Dataset A / B loaders with Excel + hardcoded fallbacks

### Frontend (Next.js 16)
- App Router, Shadcn UI, Tailwind CSS
- Drag-and-drop upload, confidence gauge, feature importance breakdown

---

## Quick Start

### Docker (recommended)
```bash
docker-compose up --build
```
Access at [http://localhost:3000](http://localhost:3000).

### Manual

**1. Place dataset files** (optional — fallback corpus used if absent):
```
backend/data/Dataset_A_Combined_Resume_Corpus.xlsx
backend/data/Dataset_B_Keyword_Heuristic_Signals.xlsx
```

**2. Backend** (Python 3.10+):
```bash
cd backend
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python setup_nltk.py              # download NLTK corpora once
uvicorn main:app --reload
```
The model trains automatically on first startup (logged as `[startup] Model trained and saved successfully.`).

**3. Frontend** (Node.js 18+):
```bash
cd frontend
npm install
npm run dev
```

### Scripts
```bash
./start_app.sh        # Mac/Linux — starts both services
start_app.bat         # Windows
```

---

## API Response

`POST /analyze/{filename}` returns:

```json
{
  "filename": "resume.pdf",
  "is_ai_generated": true,
  "label": "ai_generated",
  "confidence": 0.87,
  "explanation": ["Low perplexity (32.1)", "High AI-phrase density"],
  "matched_ai_phrases": ["proven track record", "delve into"],
  "matched_template_patterns": ["Summary", "Core Competencies"],
  "feature_importances": {
    "perplexity": 0.21,
    "ai_phrase_match_count": 0.18,
    "ttr": 0.14
  },
  "features": { "ttr": 0.61, "perplexity": 32.1, "..." : "..." },
  "raw_heuristics": { "ai_phrase_score": 0.72, "..." : "..." },
  "debug_info": {
    "extracted_text_preview": "...",
    "preprocessed_text_preview": "..."
  }
}
```

### Interpreting `confidence`

| Range | Label | Meaning |
|---|---|---|
| 0–30% | `human_written` | Likely authored by a human |
| 31–69% | `template_based` | Possibly AI-assisted or templated |
| 70–100% | `ai_generated` | Likely produced by a language model |

---

## Requirements

Full dependency list: [backend/requirements.txt](backend/requirements.txt)

Key packages: `fastapi`, `scikit-learn`, `torch`, `transformers`, `spacy`, `nltk`, `textstat`, `pandas`, `openpyxl`

See [REQUIREMENTS.md](REQUIREMENTS.md) for the full SRS and [PROJECT_QnA.md](PROJECT_QnA.md) for research methodology Q&A.
