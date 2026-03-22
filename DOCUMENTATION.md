# AI Resume Detector — Complete Project Documentation

> A research-grade full-stack web application that detects AI-generated resumes using Weakly Supervised Hybrid Learning, combining Neural Perplexity analysis, Stylometric features, and Heuristic signals.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Technology Stack](#2-technology-stack)
3. [Project Structure](#3-project-structure)
4. [System Architecture & Data Flow](#4-system-architecture--data-flow)
5. [Backend — Deep Dive](#5-backend--deep-dive)
   - 5.1 [Entry Point — main.py](#51-entry-point--mainpy)
   - 5.2 [Router: Upload](#52-router-upload)
   - 5.3 [Router: Analyze](#53-router-analyze)
   - 5.4 [Service: Text Extraction](#54-service-text-extraction)
   - 5.5 [Service: Preprocessing](#55-service-preprocessing)
   - 5.6 [Service: Weak Supervision (Heuristics)](#56-service-weak-supervision-heuristics)
   - 5.7 [Service: Feature Extraction](#57-service-feature-extraction)
   - 5.8 [Service: Hybrid ML Model](#58-service-hybrid-ml-model)
6. [Frontend — Deep Dive](#6-frontend--deep-dive)
   - 6.1 [App Layout](#61-app-layout)
   - 6.2 [Main Page — page.tsx](#62-main-page--pagetsx)
   - 6.3 [API Integration Flow](#63-api-integration-flow)
   - 6.4 [Result Rendering Logic](#64-result-rendering-logic)
7. [API Reference](#7-api-reference)
8. [ML & AI — Scientific Details](#8-ml--ai--scientific-details)
   - 8.1 [Neural Perplexity (DistilGPT2)](#81-neural-perplexity-distilgpt2)
   - 8.2 [Stylometric Analysis (Spacy)](#82-stylometric-analysis-spacy)
   - 8.3 [Weak Supervision & Heuristics](#83-weak-supervision--heuristics)
   - 8.4 [Hybrid Scoring Logic](#84-hybrid-scoring-logic)
   - 8.5 [Result Interpretation](#85-result-interpretation)
9. [Data](#9-data)
10. [Dependencies — Complete List](#10-dependencies--complete-list)
11. [Configuration & Environment](#11-configuration--environment)
12. [Docker & Deployment](#12-docker--deployment)
13. [Running Locally (Without Docker)](#13-running-locally-without-docker)
14. [Known Limitations & Future Work](#14-known-limitations--future-work)

---

## 1. Project Overview

| Property | Value |
|---|---|
| **Project Name** | AI Resume Detector |
| **Type** | Full-Stack Web Application (Research / Academic) |
| **Goal** | Classify resumes as AI-generated or Human-written |
| **AI Methods** | Neural Perplexity, Stylometrics, Weak Supervision |
| **Frontend** | Next.js 16 + React 19 + Tailwind CSS 4 |
| **Backend** | Python FastAPI + HuggingFace Transformers + Spacy |
| **Deployment** | Docker + Docker Compose |

### What it does

1. User uploads a resume file (PDF, DOCX, or TXT).
2. Backend extracts and cleans the text.
3. The system applies three analytical layers:
   - **Heuristic layer** — phrase-pattern matching and structural analysis
   - **Stylometric layer** — linguistic style statistics via Spacy NLP
   - **Neural layer** — perplexity scoring via DistilGPT2 language model
4. A hybrid scoring function combines all signals into a confidence percentage.
5. Frontend displays the verdict (AI/Human), confidence score, and human-readable reasoning.

---

## 2. Technology Stack

### Frontend

| Technology | Version | Purpose |
|---|---|---|
| Next.js | 16.1.4 | React framework with App Router (SSR/SPA) |
| React | 19.2.3 | UI component library |
| React DOM | 19.2.3 | DOM rendering |
| TypeScript | ^5 | Static type checking |
| Tailwind CSS | ^4 | Utility-first CSS styling |
| @tailwindcss/postcss | ^4 | PostCSS integration for Tailwind |
| lucide-react | ^0.562.0 | SVG icon library |
| ESLint | ^9 | Code quality linting |
| eslint-config-next | 16.1.4 | Next.js ESLint rules |

### Backend

| Technology | Version | Purpose |
|---|---|---|
| FastAPI | 0.128.0 | Async REST API framework |
| Uvicorn | 0.40.0 | ASGI server to run FastAPI |
| Python | 3.10 | Runtime language |
| PyTorch | 2.10.0 | Deep learning tensor computations |
| HuggingFace Transformers | 4.57.6 | Pre-trained NLP models (DistilGPT2) |
| Spacy | latest | NLP pipeline (POS tagging, sentence segmentation) |
| en_core_web_sm | — | Spacy English language model |
| textstat | latest | Readability and text statistics |
| scikit-learn | 1.8.0 | ML utilities |
| pypdf | 6.6.0 | PDF text extraction |
| python-docx | 1.2.0 | DOCX file parsing |
| pandas | 3.0.0 | Data analysis / CSV handling |
| numpy | 2.4.1 | Numerical array operations |
| python-multipart | latest | Multipart form data parsing for file upload |

### Infrastructure

| Technology | Purpose |
|---|---|
| Docker | Containerization of backend and frontend |
| Docker Compose | Multi-container orchestration |

---

## 3. Project Structure

```
Sweety_Project/
│
├── backend/                          # Python FastAPI backend
│   ├── main.py                       # App entry point, CORS, router registration
│   ├── requirements.txt              # Python package dependencies
│   ├── Dockerfile                    # Backend container definition
│   │
│   ├── routers/                      # HTTP route handlers (controllers)
│   │   ├── upload.py                 # POST /upload/ — saves file to disk
│   │   └── analyze.py                # POST /analyze/{filename} — runs ML pipeline
│   │
│   ├── services/                     # Core business logic (service layer)
│   │   ├── extraction.py             # Extracts raw text from PDF / DOCX / TXT
│   │   ├── preprocessing.py          # Cleans and normalizes extracted text
│   │   ├── weak_supervision.py       # Heuristic-based labeling functions
│   │   ├── features.py               # ML feature extraction (perplexity, stylometry)
│   │   └── model.py                  # Hybrid decision model (scoring + confidence)
│   │
│   ├── data/                         # Training and evaluation data
│   │   ├── ai/                       # 10 AI-generated sample resumes
│   │   ├── human/                    # 10 human-written sample resumes
│   │   └── AI_Resume_Screening.csv   # 1000+ labelled resume records
│   │
│   ├── scripts/
│   │   └── generate_data.py          # Script to generate synthetic resume data
│   │
│   └── uploads/                      # Runtime directory for uploaded files
│
├── frontend/                         # Next.js 16 web application
│   ├── app/                          # Next.js App Router
│   │   ├── layout.tsx                # Root layout — fonts, metadata, HTML shell
│   │   ├── page.tsx                  # Main page — upload UI + result display
│   │   └── globals.css               # Global Tailwind base styles
│   │
│   ├── components/
│   │   └── ui/                       # Reserved for reusable UI components (empty)
│   │
│   ├── lib/
│   │   └── utils.ts                  # cn() utility for merging Tailwind class names
│   │
│   ├── public/                       # Static assets served at /
│   │   ├── file.svg
│   │   ├── globe.svg
│   │   ├── next.svg
│   │   ├── vercel.svg
│   │   └── window.svg
│   │
│   ├── package.json                  # npm manifest + dependency list
│   ├── tsconfig.json                 # TypeScript compiler config
│   ├── next.config.ts                # Next.js framework configuration
│   └── Dockerfile                    # Frontend container definition (multi-stage)
│
├── docker-compose.yml                # Orchestrates backend + frontend containers
├── README.md                         # Project overview and quick-start guide
├── ARCHITECTURE.md                   # System architecture block diagram
├── REQUIREMENTS.md                   # Software Requirements Specification (SRS)
├── PROJECT_QnA.md                    # Research methodology Q&A document
├── RESEARCH_DIAGRAM.md               # Detailed research pipeline diagrams
├── DOCUMENTATION.md                  # This file — complete technical reference
├── start_app.sh                      # One-command startup script (Mac/Linux)
├── start_app.bat                     # One-command startup script (Windows)
└── verify_setup.py                   # Pre-flight environment verification script
```

---

## 4. System Architecture & Data Flow

### End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER (Browser)                          │
│   Uploads resume file (PDF / DOCX / TXT) via drag-drop or      │
│   file picker on the Next.js frontend at localhost:3000         │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTP POST multipart/form-data
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│             STEP 1 — FILE UPLOAD  (routers/upload.py)           │
│   POST http://localhost:8000/upload/                            │
│   • Receives file via FastAPI UploadFile                        │
│   • Saves to backend/uploads/{filename}                         │
│   • Returns: { "filename": "...", "status": "uploaded" }        │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTP POST (filename in path)
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│           STEP 2 — TEXT EXTRACTION  (services/extraction.py)    │
│   • PDF  → PdfReader (pypdf) iterates pages, joins text         │
│   • DOCX → python-docx iterates paragraphs, joins text          │
│   • TXT  → Direct file read (UTF-8)                             │
│   Output: raw_text (str)                                        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│         STEP 3 — PREPROCESSING  (services/preprocessing.py)     │
│   • Collapses all whitespace to single spaces (regex \s+)       │
│   • Removes non-printable characters                            │
│   • Returns clean_text (str)                                    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
┌─────────────────────┐   ┌───────────────────────────────────────┐
│   STEP 4 — WEAK     │   │     STEP 5 — FEATURE EXTRACTION       │
│   SUPERVISION       │   │     (services/features.py)            │
│   (services/        │   │                                       │
│   weak_supervision) │   │  Stylometric Features (textstat):     │
│                     │   │  • Flesch Reading Ease score          │
│  Heuristic Checks:  │   │  • Reading time estimate              │
│  • AI phrase scan   │   │  • Lexical Diversity (TTR)            │
│    (11+ phrases)    │   │                                       │
│  • Template struct  │   │  Neural Feature (DistilGPT2):         │
│    detection (4     │   │  • Perplexity score (sliding window)  │
│    section headers) │   │                                       │
│  • perfect_grammar  │   │  NLP Features (Spacy en_core_web_sm): │
│    (placeholder)    │   │  • Adjective ratio (ADJ / total)      │
│                     │   │  • Verb ratio (VERB / total)          │
│  Output: dict with  │   │  • Sentence length std deviation      │
│  3 boolean signals  │   │  • Sentence length mean               │
└──────────┬──────────┘   │                                       │
           │              │  Structural Feature:                  │
           │              │  • Bullet point density               │
           │              │                                       │
           │              │  Output: features dict (9 values)     │
           └──────────────┴────────────────┐
                                           ▼
┌─────────────────────────────────────────────────────────────────┐
│       STEP 6 — HYBRID ML MODEL  (services/model.py)             │
│                                                                 │
│   Weighted Score Accumulation:                                  │
│   +0.4  if uses_ai_phrases == True                              │
│   +0.2  if has_template_structure == True                       │
│   +0.2  if lexical_diversity < 0.4                              │
│   +0.1  if sent_len_std < 5 (robotic uniformity)               │
│   +0.3  if perplexity < 40  (AI-like fluency)                   │
│   -0.1  if perplexity > 80  (human-like randomness)             │
│                                                                 │
│   confidence = clamp(score, 0.0, 1.0)                          │
│   is_ai_generated = confidence > 0.5                           │
│                                                                 │
│   Output: { is_ai_generated, confidence, reasons, features }   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│         STEP 7 — RESPONSE FORMATTING  (routers/analyze.py)      │
│   Returns structured JSON:                                      │
│   {                                                             │
│     filename, is_ai_generated, confidence,                      │
│     explanation (reasons list),                                 │
│     raw_heuristics, features,                                   │
│     debug_info { extracted_text_preview,                        │
│                  preprocessed_text_preview }                    │
│   }                                                             │
└─────────────────────────┬───────────────────────────────────────┘
                          │ JSON response
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│               FRONTEND DISPLAY  (app/page.tsx)                  │
│   • Red banner + ⚠ icon  → "AI-Generated Content Detected"      │
│   • Green banner + ✓ icon → "Likely Human-Written"              │
│   • Shows Confidence Score as percentage                        │
│   • Bullet list of analysis reasons                             │
│   • Expandable "View Preprocessing Steps" debug panel           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Backend — Deep Dive

### 5.1 Entry Point — `main.py`

**File:** `backend/main.py`

```
FastAPI app
  ├── CORS Middleware (allow all origins, methods, headers)
  ├── GET /  →  health check, returns { "message": "...running" }
  ├── Router: upload   (prefix: /upload)
  └── Router: analyze  (prefix: /analyze)
```

- CORS is open (`allow_origins=["*"]`) — suitable for local development
- Uvicorn runs the app on port **8000**
- Routers are registered via `app.include_router()`

---

### 5.2 Router: Upload

**File:** `backend/routers/upload.py`
**Endpoint:** `POST /upload/`

**Request:**
- `Content-Type: multipart/form-data`
- Form field: `file` (UploadFile)

**Process:**
1. Accepts any file (no type validation at router level)
2. Creates `uploads/` directory if it doesn't exist
3. Streams file bytes to `uploads/{original_filename}` using `shutil.copyfileobj`

**Response:**
```json
{ "filename": "resume.pdf", "status": "uploaded" }
```

**Error Handling:**
- HTTP 500 on any exception during file write

---

### 5.3 Router: Analyze

**File:** `backend/routers/analyze.py`
**Endpoint:** `POST /analyze/{filename}`

**Process (sequential pipeline):**
1. Verifies file exists in `uploads/` — raises HTTP 404 if not
2. Calls `TextExtractor.extract_from_file(file_path)`
3. Calls `TextPreprocessor.clean_text(text)`
4. Calls `WeakSupervision.apply_heuristics(clean_text)`
5. Calls `model.predict(clean_text, heuristics)`
6. Constructs and returns the full JSON response

**Response Schema:**
```json
{
  "filename": "resume.pdf",
  "is_ai_generated": true,
  "confidence": 0.7,
  "explanation": ["Contains common AI-generated phrases.", "..."],
  "raw_heuristics": {
    "uses_ai_phrases": true,
    "has_template_structure": false,
    "perfect_grammar": false
  },
  "features": {
    "flesch_reading_ease": 45.2,
    "reading_time": 3.1,
    "lexical_diversity": 0.38,
    "perplexity": 27.5,
    "adj_ratio": 0.06,
    "verb_ratio": 0.09,
    "sent_len_std": 3.8,
    "sent_len_mean": 18.2,
    "bullet_density": 0.12
  },
  "debug_info": {
    "extracted_text_preview": "John Doe...(first 500 chars)",
    "preprocessed_text_preview": "John Doe...(first 500 chars, cleaned)"
  }
}
```

> **Bug Note:** `"explanation"` key is duplicated in the response dict — Python will silently use the last assignment. Both values are `result["reasons"]`, so output is correct.

---

### 5.4 Service: Text Extraction

**File:** `backend/services/extraction.py`

**Class:** `TextExtractor`

| Method | Input | Output | Library |
|---|---|---|---|
| `extract_from_file(file_path)` | File path string | Text string | Routes to below |
| `extract_from_pdf(file_path)` | PDF path | Joined page text | pypdf PdfReader |
| `extract_from_docx(file_path)` | DOCX path | Joined paragraph text | python-docx Document |
| Direct read | TXT path | Raw file content | Built-in open() |

**Supported Extensions:** `.pdf`, `.docx`, `.txt`
**Unsupported:** Raises `ValueError` for anything else

---

### 5.5 Service: Preprocessing

**File:** `backend/services/preprocessing.py`

**Class:** `TextPreprocessor`

| Method | What it does |
|---|---|
| `clean_text(text)` | Collapses whitespace with `re.sub(r'\s+', ' ')`, strips, removes non-printable chars |
| `segment_text(text)` | Placeholder — returns `{"full_text": text, "segments": {}}` (not yet implemented) |

**Why preprocessing matters:**
- Raw PDF text often contains double/triple spaces, line breaks, and control characters
- Clean text ensures consistent token counts and accurate NLP analysis

---

### 5.6 Service: Weak Supervision (Heuristics)

**File:** `backend/services/weak_supervision.py`

**Class:** `WeakSupervision`

#### Heuristic 1 — AI Phrase Detection
Checks if any of 11 known AI-ism phrases appear in the lowercased text:

| Phrase | Meaning |
|---|---|
| "delve into" | Very common in ChatGPT output |
| "testament to" | Over-formal, hallmark of AI writing |
| "landscape of" | Vague, corporate AI phrasing |
| "meticulously crafted" | AI self-congratulatory phrasing |
| "tapestry of" | AI metaphor overuse |
| "underscores the importance" | AI academic hedging |
| "poised to" | AI future-tense framing |
| "leveraging the power of" | AI buzzword stacking |
| "realm of" | AI abstract domain reference |
| "navigate the complexities" | AI problem-description template |
| "foster a culture of" | AI leadership/HR boilerplate |

Returns `True` if **any** phrase found (OR logic).

#### Heuristic 2 — Template Structure Detection
Looks for all 4 canonical resume sections:
1. "professional summary" OR "summary" OR "objective"
2. "skills" OR "core competencies" OR "technical skills"
3. "experience" OR "work history" OR "employment"
4. "education" OR "academic background"

Returns `True` only if **all 4** are present (AND logic). This is a **weak signal** — human resumes also use standard sections.

#### Heuristic 3 — Perfect Grammar
`perfect_grammar` is always `False` — placeholder for future integration of a grammar checker library.

---

### 5.7 Service: Feature Extraction

**File:** `backend/services/features.py`

**Class:** `FeatureExtractor`

Models loaded at module import time (singleton pattern):
- `spacy.load("en_core_web_sm")` — English NLP pipeline
- `GPT2TokenizerFast.from_pretrained("distilgpt2")` — tokenizer
- `GPT2LMHeadModel.from_pretrained("distilgpt2")` — language model (eval mode)

Both models have graceful `try/except` fallbacks (return `None` if unavailable).

#### Feature Breakdown

| Feature | Library | How Computed | AI Signal |
|---|---|---|---|
| `flesch_reading_ease` | textstat | Standard readability formula | Low score = complex text |
| `reading_time` | textstat | Word count / average reading speed | Informational only |
| `lexical_diversity` | Custom | `len(set(tokens)) / len(tokens)` | Low (<0.4) = repetitive = AI |
| `perplexity` | DistilGPT2 | Cross-entropy loss, exponentiated | Low (<40) = AI fluency |
| `adj_ratio` | Spacy | ADJ token count / total tokens | Informational |
| `verb_ratio` | Spacy | VERB token count / total tokens | Informational |
| `sent_len_std` | Spacy | Std dev of sentence lengths | Low (<5) = robotic uniformity |
| `sent_len_mean` | Spacy | Mean sentence length | Informational |
| `bullet_density` | Custom | `(count("•") + count("-")) / lines` | Structural signal |

#### Perplexity Calculation (Sliding Window)

The DistilGPT2 perplexity uses a proper sliding window approach to handle texts longer than the model's context window (1024 tokens):

```python
stride = 512
for begin_loc in range(0, seq_len, stride):
    end_loc = min(begin_loc + max_length, seq_len)
    # compute NLL for target tokens
    nlls.append(neg_log_likelihood)

ppl = exp(mean(nlls))
```

This is the standard HuggingFace perplexity evaluation method, preventing context truncation artifacts.

---

### 5.8 Service: Hybrid ML Model

**File:** `backend/services/model.py`

**Class:** `ResumeModel`

The `predict(text, heuristics)` method implements a **rule-based weighted scoring system** — not a trained ML classifier. This is an intentional design choice for explainability and zero-shot operation.

#### Scoring Table

| Condition | Score Delta | Reason Added to Explanation |
|---|---|---|
| `uses_ai_phrases == True` | +0.40 | "Contains common AI-generated phrases." |
| `has_template_structure == True` | +0.20 | "Follows a generic AI/Template structure." |
| `lexical_diversity < 0.4` | +0.20 | "Low lexical diversity (repetitive vocabulary)." |
| `sent_len_std < 5` | +0.10 | "Very uniform sentence lengths (robotic flow)." |
| `40 > perplexity > 0` | +0.30 | "Low Perplexity (N): Indicative of AI text." |
| `perplexity > 80` | -0.10 | "High Perplexity (N): Indicative of Human text." |

**Maximum possible score:** 1.20 (before clamping)
**After clamping:** `confidence = min(max(score, 0.0), 1.0)`
**Classification threshold:** `confidence > 0.5` → AI-generated

---

## 6. Frontend — Deep Dive

### 6.1 App Layout

**File:** `frontend/app/layout.tsx`

- Sets HTML `<html lang="en">`
- Loads **Geist** and **Geist_Mono** fonts from `next/font/google`
- Defines metadata: title `"Create Next App"`, description placeholder
- Wraps entire app in font CSS variables

### 6.2 Main Page — `page.tsx`

**File:** `frontend/app/page.tsx`

Marked `"use client"` — runs entirely in the browser as a React Client Component.

#### State Variables

| State | Type | Initial | Purpose |
|---|---|---|---|
| `file` | `File \| null` | `null` | Holds the selected resume file |
| `analyzing` | `boolean` | `false` | Controls loading/disabled state |
| `result` | `any \| null` | `null` | Stores the full API response |

#### Event Handlers

**`handleFileChange(e)`**
- Triggered by `<input type="file">` change event
- Sets `file` state to the first selected file
- Resets `result` to `null` (clears previous analysis)

**`handleAnalyze()`**
- Guards: returns early if no file selected
- Sets `analyzing = true` (shows spinner)
- Step 1: `POST http://localhost:8000/upload/` with `FormData`
- Step 2: `POST http://localhost:8000/analyze/{file.name}`
- Sets `result` with the parsed JSON response
- Sets `analyzing = false` on both success and error
- On error: `alert("Analysis failed. Make sure backend is running.")`

#### UI Sections

1. **Header** — "Resume AI Detector" h1 title
2. **Upload Card** — white card with shadow containing:
   - Dashed border drop zone with invisible `<input type="file">` overlay
   - Shows filename when selected, placeholder text when empty
   - Upload icon + supported formats label
3. **Analyze Button** — full-width blue button, disabled while analyzing or no file
4. **Result Panel** — conditionally rendered when `result` is not null:
   - Red background (`bg-red-50`) for AI-detected
   - Green background (`bg-green-50`) for human
   - `AlertTriangle` or `CheckCircle` icon
   - Verdict headline
   - Confidence score (multiplied by 100, 1 decimal)
   - Bulleted explanation list
   - Expandable `<details>` panel with raw extracted text and cleaned text previews

---

### 6.3 API Integration Flow

```
Frontend (page.tsx)
  │
  ├── 1. POST /upload/
  │      Body: FormData { file: File }
  │      Success: { filename, status }
  │      Failure: throws Error("Upload failed")
  │
  └── 2. POST /analyze/{file.name}
         Body: empty (filename from URL path)
         Success: full analysis JSON object → stored in result state
         Failure: caught in catch block → alert shown
```

**Hard-coded base URL:** `http://localhost:8000`
No environment variable abstraction — development-only configuration.

---

### 6.4 Result Rendering Logic

```typescript
result.is_ai_generated
  ? Red panel + ⚠ AlertTriangle + "AI-Generated Content Detected"
  : Green panel + ✓ CheckCircle + "Likely Human-Written"

Confidence: (result.confidence * 100).toFixed(1) + "%"

Explanation: result.explanation.map(reason => <li>{reason}</li>)

Debug panel (collapsible <details>):
  - result.debug_info.extracted_text_preview
  - result.debug_info.preprocessed_text_preview
```

---

## 7. API Reference

### Base URL
```
http://localhost:8000
```

### Endpoints

#### `GET /`
Health check.

**Response:**
```json
{ "message": "Resume AI Detector API is running" }
```

---

#### `POST /upload/`
Upload a resume file.

**Request:**
```
Content-Type: multipart/form-data
Field: file  (PDF, DOCX, or TXT)
```

**Success Response (200):**
```json
{ "filename": "john_doe_resume.pdf", "status": "uploaded" }
```

**Error Response (500):**
```json
{ "detail": "<error message>" }
```

---

#### `POST /analyze/{filename}`
Run the AI detection pipeline on an uploaded file.

**Path Parameter:** `filename` — the filename returned from `/upload/`

**Success Response (200):**
```json
{
  "filename": "john_doe_resume.pdf",
  "is_ai_generated": true,
  "confidence": 0.7,
  "explanation": [
    "Contains common AI-generated phrases.",
    "Low Perplexity (28): Indicative of AI text."
  ],
  "raw_heuristics": {
    "uses_ai_phrases": true,
    "has_template_structure": false,
    "perfect_grammar": false
  },
  "features": {
    "flesch_reading_ease": 52.1,
    "reading_time": 2.8,
    "lexical_diversity": 0.35,
    "perplexity": 28.4,
    "adj_ratio": 0.07,
    "verb_ratio": 0.10,
    "sent_len_std": 4.2,
    "sent_len_mean": 15.6,
    "bullet_density": 0.08
  },
  "debug_info": {
    "extracted_text_preview": "John Doe | Software Engineer...",
    "preprocessed_text_preview": "John Doe | Software Engineer..."
  }
}
```

**Error Responses:**
- `404` — File not found in uploads directory
- `400` — Text extraction failed (corrupted file)

---

## 8. ML & AI — Scientific Details

### 8.1 Neural Perplexity (DistilGPT2)

**Model:** `distilgpt2` — a distilled (smaller, faster) version of GPT-2 from HuggingFace.

**What is perplexity?**
Perplexity measures how "surprised" a language model is by a given text. It is the exponentiated average negative log-likelihood per token:

```
PPL = exp( (1/N) * Σ -log P(token_i | context) )
```

**Why it works for AI detection:**
- AI-generated text is produced by a language model optimizing for high probability (low surprise) completions
- This makes AI text statistically "easy" for another LM to predict → **low perplexity**
- Human-written text has more unpredictable word choices, slang, personal style → **high perplexity**

**Thresholds used:**
| Perplexity Range | Interpretation | Score Effect |
|---|---|---|
| < 40 | Very fluent, likely AI | +0.30 |
| 40 – 80 | Neutral zone | No effect |
| > 80 | Unpredictable, likely human | -0.10 |

**Limitation:** DistilGPT2 was trained on general web text (WebText), not specifically resumes. Perplexity values vary by text length and domain.

---

### 8.2 Stylometric Analysis (Spacy)

Stylometry is the statistical analysis of writing style. The features extracted are:

**Lexical Diversity (Type-Token Ratio)**
```
TTR = unique_words / total_words
```
- AI models tend to reuse the same vocabulary repeatedly across all generated content
- Low TTR (<0.4) is a weak indicator of AI text
- Human experts naturally vary their word choices

**Sentence Length Std Dev**
- AI-generated text tends to produce sentences of very consistent length
- High variability in sentence length is a human writing characteristic
- `sent_len_std < 5` (very uniform) adds +0.10 to the AI score

**POS Ratios (Adjective & Verb)**
- Spacy tags each token with its part-of-speech
- AI writing tends to be noun-heavy with decorative adjective overuse
- These ratios are extracted but not currently used in the scoring function (collected for future analysis)

---

### 8.3 Weak Supervision & Heuristics

**Weak Supervision** is a machine learning paradigm where noisy, heuristic labeling functions replace expensive human annotations. In this system:

- **Labeling Functions (LFs)** are the individual heuristic checks
- Each LF produces a signal: +1 (AI), -1 (Human), or 0 (Abstain)
- The hybrid model acts as an aggregation of these weak signals

**Why "weak" supervision?**
- No large labeled training dataset was needed
- Domain knowledge about AI writing patterns is encoded directly as rules
- The system is immediately deployable without a training phase

---

### 8.4 Hybrid Scoring Logic

The final confidence score is a **weighted linear combination** of binary signal triggers:

```
score = 0
if uses_ai_phrases:   score += 0.40
if template_struct:   score += 0.20
if TTR < 0.4:         score += 0.20
if sent_std < 5:      score += 0.10
if ppl < 40:          score += 0.30
if ppl > 80:          score -= 0.10

confidence = clamp(score, 0.0, 1.0)
```

**Weight rationale:**
- `uses_ai_phrases` (0.40) — Strongest signal; specific phrases are highly correlated with AI output
- `perplexity < 40` (0.30) — Strong neural signal; low perplexity is mathematically linked to LM generation
- `lexical_diversity` (0.20) — Moderate signal; can also apply to specialists writing in narrow domains
- `template_structure` (0.20) — Weak signal; humans also use standard resume formats
- `sent_len_std` (0.10) — Supplementary signal; weakest individual indicator

---

### 8.5 Result Interpretation

| Confidence % | Verdict | Meaning |
|---|---|---|
| 0% – 30% | Likely Human | High perplexity, diverse vocabulary, natural sentence variation, no AI phrases |
| 30% – 50% | Uncertain (Human leaning) | Mixed signals, not enough AI indicators |
| 50% – 70% | AI Suspected | Multiple weak signals triggered |
| 70% – 100% | AI Detected | Strong signals: AI phrases + low perplexity + repetitive vocabulary |

---

## 9. Data

### Sample Data

| Location | Contents | Purpose |
|---|---|---|
| `backend/data/ai/` | 10 AI-generated resume text files | Reference/testing |
| `backend/data/human/` | 10 human-written resume text files | Reference/testing |

### Dataset

**File:** `backend/data/AI_Resume_Screening.csv` (tracked in root git status)

- 1000+ resume records
- Contains AI-likelihood scores and metadata
- Used for reference analysis and potential future model training
- **Not actively used during inference** — the current pipeline is zero-shot

### Data Generation

**File:** `backend/scripts/generate_data.py`

Script for generating synthetic resume data for expanding the training corpus.

---

## 10. Dependencies — Complete List

### Frontend (`frontend/package.json`)

```json
{
  "dependencies": {
    "next": "16.1.4",
    "react": "19.2.3",
    "react-dom": "19.2.3",
    "lucide-react": "^0.562.0"
  },
  "devDependencies": {
    "@tailwindcss/postcss": "^4",
    "tailwindcss": "^4",
    "typescript": "^5",
    "@types/react": "^19",
    "@types/react-dom": "^19",
    "@types/node": "^20",
    "eslint": "^9",
    "eslint-config-next": "16.1.4"
  }
}
```

### Backend (`backend/requirements.txt`)

```
fastapi==0.128.0
uvicorn==0.40.0
python-multipart
torch==2.10.0
transformers==4.57.6
spacy
textstat
scikit-learn==1.8.0
pypdf==6.6.0
python-docx==1.2.0
pandas==3.0.0
numpy==2.4.1
```

**Additional runtime download:**
```bash
python -m spacy download en_core_web_sm
```

---

## 11. Configuration & Environment

### TypeScript (`frontend/tsconfig.json`)

```json
{
  "compilerOptions": {
    "target": "ES2017",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [{ "name": "next" }],
    "paths": { "@/*": ["./*"] }
  }
}
```

### Environment Variables
Currently **none defined**. All URLs are hard-coded:
- Frontend calls: `http://localhost:8000`
- Backend runs on: `0.0.0.0:8000`
- Frontend serves on: `0.0.0.0:3000`

### CORS Policy
Backend allows all origins in development:
```python
allow_origins=["*"]
allow_credentials=True
allow_methods=["*"]
allow_headers=["*"]
```

---

## 12. Docker & Deployment

### Backend Dockerfile (`backend/Dockerfile`)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
# Install system deps, copy requirements, pip install
# Download spacy model: python -m spacy download en_core_web_sm
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Frontend Dockerfile (`frontend/Dockerfile`)

Multi-stage build:
```dockerfile
# Stage 1 — Builder
FROM node:18-alpine AS builder
# npm ci, npm run build

# Stage 2 — Runner
FROM node:18-alpine AS runner
# Copy built artifacts, set NODE_ENV=production
EXPOSE 3000
CMD ["npm", "start"]
```

### Docker Compose (`docker-compose.yml`)

```yaml
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    volumes:
      - ./backend/uploads:/app/uploads
      - ./backend/data:/app/data
    restart: always

  frontend:
    build: ./frontend
    ports: ["3000:3000"]
    depends_on: [backend]
    restart: always
```

### Deploy Commands

```bash
# Build and start all services
docker-compose up --build

# Start in background
docker-compose up -d --build

# Stop all services
docker-compose down

# View logs
docker-compose logs -f
```

---

## 13. Running Locally (Without Docker)

### One-Command Start

**Mac/Linux:**
```bash
chmod +x start_app.sh
./start_app.sh
```

**Windows:**
```bat
start_app.bat
```

### Manual Setup

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate       # Mac/Linux
# OR: venv\Scripts\activate    # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn main:app --reload --port 8000
```

**Frontend (new terminal):**
```bash
cd frontend
npm install
npm run dev
```

**Access:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs (Swagger): http://localhost:8000/docs
- API Docs (ReDoc): http://localhost:8000/redoc

### Verify Setup
```bash
python verify_setup.py
```

---

## 14. Known Limitations & Future Work

### Current Limitations

| Area | Limitation |
|---|---|
| Authentication | None — fully public API |
| File Validation | No server-side MIME type checking; relies on file extension only |
| Storage | Files accumulate in `uploads/` with no cleanup mechanism |
| Model | DistilGPT2 is a small, general model — not fine-tuned on resume data |
| Phrase List | Only 11 AI phrases — easily bypassed by paraphrasing |
| `perfect_grammar` | Placeholder, always returns False |
| `segment_text` | Placeholder, not wired into the pipeline |
| Duplicate Key | `"explanation"` key is duplicated in `analyze.py` response dict |
| Environment | No `.env` file support; URLs are hard-coded |
| Testing | No unit tests, integration tests, or CI/CD pipeline |
| Frontend | Single page, no routing, no component library built out |

### Suggested Improvements

1. **Fine-tune model** on a labeled resume corpus for domain-specific perplexity
2. **Expand AI phrase list** to 50+ verified phrases with weighted scoring
3. **Add grammar checking** via LanguageTool API or similar
4. **Implement `segment_text`** to analyze resume sections independently
5. **Add file type validation** using python-magic (MIME detection)
6. **Implement upload cleanup** via background task or scheduled job
7. **Add environment variables** via `python-dotenv` and `next/env`
8. **Write tests** — pytest for backend services, Jest/Vitest for frontend
9. **Add rate limiting** via FastAPI middleware
10. **Build out component library** in `frontend/components/ui/`

---

*Documentation generated: 2026-03-23*
*Project: AI Resume Detector — Research Grade Full-Stack Application*
