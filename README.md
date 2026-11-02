# AI Resume Detector

Classifies a resume as **human-written**, **AI-generated**, or **template-based** using a weakly supervised hybrid pipeline: heuristic phrase signals + stylometric features + a Random Forest / XGBoost ensemble, with a human-readable explanation for every decision.

| | |
|---|---|
| **Backend** | Python 3.12 · FastAPI · scikit-learn · XGBoost · NLTK · textstat · (optional) spaCy, DistilGPT2 |
| **Frontend** | Next.js 16 · React 19 · Tailwind CSS 4 · lucide-react |
| **Input** | PDF, DOCX, TXT (max 10 MB) |
| **Output** | 3-class label, confidence, per-model probabilities, feature values, top-5 feature importances, reasons |

---

## Quick start

### Docker (recommended: works on any Mac, Intel or Apple Silicon)
1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and open it once so the engine is running.
2. Copy the project folder to the Mac (or `git clone` it).
3. From the project folder:
   ```bash
   docker compose up --build -d
   ```
   The first build downloads about 3 GB and takes roughly 10 minutes. Later starts take seconds. The frontend waits until the API reports healthy.
4. Open http://localhost:3000 (API: http://localhost:8000).

| Task | Command |
|---|---|
| Stop | `docker compose down` |
| Start again | `docker compose up -d` |
| Logs | `docker compose logs -f backend` |
| Rebuild after code changes | `docker compose up --build -d` |

Uploaded files are kept in `backend/uploads/` on the host. Python, Node and XGBoost don't need to be installed on the Mac; everything runs inside the containers.

### Manual
```bash
# Backend (Python 3.12+)
cd backend
python3 -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python setup_nltk.py                                  # NLTK corpora, once
python -m spacy download en_core_web_sm               # optional, POS explanations only
uvicorn main:app --reload --port 8000

# Frontend (Node 20.9+)
cd frontend
npm install
npm run dev
```
Or start both with `./start_app.sh` (macOS/Linux) or `start_app.bat` (Windows).

**Check the install:** `python verify_setup.py` (from the repo root, with the backend venv active). It tests phrase detection, model loading, prediction, and that path traversal is blocked.

---

## How it works

```mermaid
graph TD
    U(("User")) -->|PDF / DOCX / TXT| FE["Frontend<br/>frontend/app/page.tsx"]
    FE -->|POST /upload/| UP["Upload<br/>routers/upload.py"]
    FE -->|POST /analyze/{filename}| AN["Analyze<br/>routers/analyze.py"]
    AN --> EX["1. Extraction<br/>services/extraction.py"]
    EX --> PP["2. Preprocessing<br/>services/preprocessing.py"]
    PP --> WS["3. Weak supervision<br/>services/weak_supervision.py"]
    PP --> FX["Feature extraction<br/>services/features.py"]
    DB[("Dataset B<br/>50 AI phrases · 24 template patterns")] --> WS
    WS -->|phrase counts + scores| ML
    FX -->|19 model features| ML["4. RF + XGBoost ensemble<br/>services/model.py"]
    DA[("Dataset A / backend/data/{human,ai}")] --> TR["ModelTrainer"] --> M[("backend/models/")] --> ML
    ML -->|label · confidence · reasons| FE
```

### 1. Extraction
`pypdf` (layout mode, so bullets and line breaks survive) for PDF, `python-docx` for DOCX including table cells, and plain read for TXT.

### 2. Preprocessing
Removes non-printable characters, normalises line endings, collapses repeated spaces and blank lines. Case, punctuation and line structure are **kept**, because bullet, section and all-caps features depend on them.

### 3. Weak supervision (Dataset B)
Programmatic labelling functions instead of hand labels:
- **50 AI signature phrases** (e.g. *"delve into"*, *"proven track record"*, *"leverage the power of"*) and **24 template patterns** (e.g. *"[Company Name]"*, *"references available upon request"*, *"I hereby declare"*), each weighted Very High = 1.0, High = 0.75, Medium = 0.5, Low = 0.25.
- Score = sum of matched weights ÷ (number of phrases × 0.5), capped at 1.
- Emits `ai_phrase_score`, `template_score`, matched phrase lists, and a `weak_label` (`ai_generated` / `template_based` / `uncertain`).
- Source: `backend/data/Dataset_B_Keyword_Heuristic_Signals.xlsx` if present, otherwise the embedded lists in `backend/data/load_datasets.py`.

### 4. Features
23 features are computed per resume. **19 feed the classifier:**

| Group | Features |
|---|---|
| Lexical | type-token ratio, avg word length, function-word ratio, comma density, passive-voice ratio |
| Rhythm | avg sentence length, sentence-length std ("burstiness") |
| Structure | bullet count, avg bullet word count, bullet length variance, section count, all-caps ratio |
| Readability | Flesch reading ease, Gunning Fog, reading time |
| Informality | first-person pronoun count, contraction count |
| Heuristic | AI phrase match count, template pattern match count |

**4 are explanation-only:** `perplexity` (DistilGPT2), `noun_density`, `adjective_density` and `verb_density` (spaCy). These libraries are optional and skipped during training, so feeding them to the model would mean training on zeros and predicting on real values. They still appear in the UI and the reasons. Perplexity is computed only when you call `/analyze/{file}?skip_perplexity=false`.

### 5. Classification
- **Random Forest** (200 trees, balanced class weights) and **XGBoost** (200 trees, depth 4, lr 0.1) are trained on the same 19 features.
- **Soft vote:** the two models' class probabilities are averaged.
- **Heuristic calibration:** strong phrase evidence (≥ 3 AI phrases or AI score ≥ 0.20) shifts probability toward `ai_generated`. Strong template evidence (≥ 2 patterns or score ≥ 0.20) shifts it toward `template_based`. When nothing matches, there is no adjustment: the absence of known phrases is not evidence of human writing.
- The label is the class with the highest probability, and `confidence` is that probability. `is_ai_generated` is true for both `ai_generated` and `template_based`.
- If both models fail to load, a rule-based scorer answers instead (`model_results.ensemble.method = "rule_based"`).

| Class | Int | Meaning |
|---|---|---|
| `human_written` | 0 | Authored by a person |
| `ai_generated` | 1 | Produced by a language model |
| `template_based` | 2 | Filled-in rigid template / boilerplate CV |

### Training data
`ModelTrainer` (in `services/model.py`) uses the first source it finds:
1. `backend/data/Dataset_A_Combined_Resume_Corpus.xlsx` (needs `resume_text` + `label` columns, ≥ 10 rows), **or**
2. `backend/data/human/*.txt` and `backend/data/ai/*.txt`, labelled **by folder**, plus 4 embedded human tech resumes and 4 embedded template resumes (28 samples today, generated with `backend/scripts/generate_data.py`).

User uploads are **never** used for training.

Trained models are committed in `backend/models/`. To retrain, run `python train_models.py` in `backend/`; it prints k-fold cross-validation accuracy. If `backend/models/` is missing, the server trains automatically on startup.

Current result: **4-fold CV accuracy 0.964 ± 0.062 on 28 samples.** Read the limitations below before quoting it.

---

## API

### `GET /`
Health check: `{"message": "Resume AI Detector API is running"}`

### `POST /upload/`
Multipart field `file`. Accepts `.pdf`, `.docx`, `.txt` up to 10 MB (larger files get a 413). The file is saved with a random prefix so uploads with the same name don't overwrite each other.
```json
{"filename": "3f9a1c2e_resume.pdf", "status": "uploaded"}
```

### `POST /analyze/{filename}?skip_perplexity=true`
`filename` is the name returned by `/upload/`, or a bundled sample such as `AI_Candidate_1.txt`. Only bare filenames are accepted; paths are rejected.
```json
{
  "filename": "3f9a1c2e_resume.pdf",
  "is_ai_generated": true,
  "label": "ai_generated",
  "confidence": 0.82,
  "explanation": ["Matched AI phrases: \"proven track record\", \"delve into\"."],
  "matched_ai_phrases": ["proven track record", "delve into"],
  "matched_template_patterns": [],
  "raw_heuristics": {"ai_phrase_score": 0.06, "weak_label": "uncertain", "...": "..."},
  "features": {"type_token_ratio": 0.61, "perplexity": 0.0, "...": "..."},
  "feature_importances": {"template_pattern_match_count": 0.12, "...": "..."},
  "model_results": {
    "random_forest": {"label": "...", "confidence": 0.0, "probabilities": {}},
    "xgboost":       {"label": "...", "confidence": 0.0, "probabilities": {}, "available": true},
    "ensemble":      {"label": "...", "confidence": 0.0, "method": "soft_vote", "probabilities": {}}
  },
  "debug_info": {"extracted_text_preview": "...", "preprocessed_text_preview": "..."}
}
```

---

## Project structure
```
backend/
  main.py                 FastAPI app, CORS (localhost:3000)
  routers/upload.py       validated, size-capped upload
  routers/analyze.py      4-step pipeline endpoint
  services/extraction.py  PDF / DOCX / TXT → text
  services/preprocessing.py
  services/weak_supervision.py
  services/features.py
  services/model.py       ModelTrainer + ResumeModel
  data/load_datasets.py   Dataset A/B loaders + embedded vocabularies
  data/{human,ai}/        labelled sample corpus
  models/                 rf_model.pkl, xgb_model.json, feature_names.json
  scripts/generate_data.py
  train_models.py         retrain + print CV accuracy
  setup_nltk.py
frontend/app/page.tsx     upload UI, result dashboard, Quick Test samples
docker-compose.yml
verify_setup.py           end-to-end self-check
```
`backend/uploads/` holds user files at runtime. It is git-ignored because resumes contain personal data.

---

## Design Q&A

**Is the data labelled?** Not at inference time. For training, the AI-phrase and template rules act as *labelling functions*: noisy, programmatic signals that replace expensive hand labelling (weak supervision). The current corpus is labelled by folder.

**Why hybrid?** The heuristics are precise but easy to evade by paraphrasing. The statistical features generalise but are noisy on small data. The ensemble combines both, and the feature vector is the bridge between them. Optional neural perplexity (DistilGPT2) adds a deep-learning signal to the explanations.

**Why Random Forest + XGBoost?** Both work well on small tabular feature sets, expose feature importances for explainability, and train in seconds. Averaging their probabilities reduces the variance of either model alone.

**Why perplexity?** Language-model text is statistically predictable (low perplexity), while human writing is more "bursty". It is currently explanation-only; see the limitations below.

**How is it explainable?** Every response lists the threshold-based reasons, the matched phrases, the raw feature values, the top-5 Random Forest importances, and each model's probabilities.

---

## Requirements
- **Hardware:** 8 GB RAM (16 GB recommended if perplexity is enabled), about 5 GB disk. Apple Silicon or x86.
- **Software:** Python 3.12+, Node 20.9+, or Docker + Docker Compose. Internet access on first run to download NLTK data, the spaCy model and DistilGPT2.
- **Performance target:** under 10 s per 2-page resume (perplexity off).

---

## Known limitations
| Area | Limitation |
|---|---|
| Training data | 28 short, script-generated samples. They are easy to separate, so the 0.96 CV score **overstates real-world accuracy**. Real labelled resumes (Dataset A) are needed before quoting any accuracy. |
| Evasion | AI text that avoids the 50 listed phrases is often classified human. |
| Perplexity / POS | Not model inputs until training also computes them. |
| Uploads | Never deleted automatically. |
| Security | No authentication or rate limiting. Intended for local / demo use. |
| Tests | `verify_setup.py` is the only automated check. |
