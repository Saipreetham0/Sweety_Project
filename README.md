# Research-Grade AI Resume Detector

A scientifically rigorous system designed to detect AI-generated resumes (e.g., ChatGPT, Claude) using **Weakly Supervised Hybrid Learning**. This project implements advanced **Stylometric**, **Semantic**, and **Structural** analysis techniques suitable for academic research.

## 🧠 Scientific Methodology

This system goes beyond simple keyword matching by implementing a multi-dimensional feature extraction pipeline:

### 1. Neural Perplexity (The "Surprise" Factor)
-   **Concept**: AI models are trained to maximize probability, resulting in text that is statistically "predictable." Human writing is more chaotic (bursty).
-   **Implementation**: We use a pre-trained **GPT-2 (DistilGPT2)** model (`transformers` library) to calculate the perplexity of the resume text.
-   **Thresholds**:
    -   **Low Perplexity (< 40)**: Strong indicator of AI generation (smooth, predictable text).
    -   **High Perplexity (> 80)**: Strong indicator of Human writing (complex, varied text).

### 2. Stylometric Analysis (Linguistic Fingerprint)
-   **Lexical Diversity (Type-Token Ratio)**: Calculates the richness of vocabulary. AI tends to be repetitive.
-   **Sentence Length Standard Deviation**: Humans vary their sentence lengths (short, then long). AI text often has a robotic, uniform rhythm (low standard deviation).
-   **POS Distribution**: Analysis of Adjective/Verb ratios using **Spacy** NLP.

### 3. Structural & Heuristic Detection (Weak Supervision)
-   **Template Recognition**: Detects standard "AI Layouts" (e.g., *Summary → Skills → Experience → Education* in that exact order).
-   **AI-Isms**: Scans for 50+ known AI hallucinations and overused phrases (e.g., *"delve into"*, *"tapestry of"*, *"meticulously crafted"*).

---

## 🏗 Technical Architecture

For a detailed visual representation of the system components and data flow, please see the [Architecture Diagram](ARCHITECTURE.md).


### **Backend (Python / FastAPI)**
The analysis engine is built on a modular microservices architecture:
-   **Core**: `FastAPI` for high-performance async API.
-   **NLP Engine**: `Spacy` (en_core_web_sm) for linguistic parsing.
-   **ML Inference**: `PyTorch` + `HuggingFace Transformers` for the GPT-2 Perplexity model.
-   **Feature Engineering**: `Textstat` for readability metrics.

### **Frontend (Next.js 16)**
-   **Framework**: Next.js 16 (App Router) for Server-Side Rendering.
-   **UI Library**: Shadcn UI + Tailwind CSS for a professional, responsive dashboard.
-   **Visualization**: Real-time display of Confidence Scores and Explanation factors.

### **Deployment (Docker)**
-   **Containerization**: Full support for Docker & Docker Compose to ensure identical environments (Python 3.10 + Node 18) across all machines.
-   **Reproducibility**: `backend/Dockerfile` locks all ML dependencies including Spacy models.

---

## 🚀 Quick Start

### 1. One-Click Launch (Docker) - Recommended for New Systems
This handles all dependencies (Python, Node, Spacy) automatically.
```bash
docker-compose up --build
```
*Access dashboard at [http://localhost:3000](http://localhost:3000).*

### 2. Standard Launch (Local Script)
**Mac/Linux:**
```bash
./start_app.sh
```

**Windows:**
Double-click `start_app.bat` or run:
```cmd
start_app.bat
```
*(This will open two new command windows for Backend and Frontend. Close them to stop.)*

### 3. Manual Installation
**Backend** (Requires Python 3.10+):
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn main:app --reload
```

**Frontend** (Requires Node.js 18+):
```bash
cd frontend
npm install
npm run dev
```

---

## 📊 Interpreting Results

When you analyze a resume, the system provides a **Confidence Score (0% - 100%)**:

-   **0% - 30% (Likely Human)**:
    -   High Perplexity (>80).
    -   High Lexical Diversity.
    -   Varied sentence structures.
-   **70% - 100% (Likely AI)**:
    -   Low Perplexity (<40).
    -   Contains "AI-isms" ("delve into").
    -   Follows a rigid, generic template structure.

---

## 🔮 Future Research Directions
-   **Adversarial Training**: Train the model against paraphrased AI text (e.g., Quillbot).
-   **Visual Layout Analysis**: Use Computer Vision (ResNet) to detect visual template matches.
-   **Vector Embeddings**: Use `all-MiniLM-L6-v2` to cluster resumes by semantic content.

---

## 📋 Requirements
For a detailed list of hardware, software, and functional requirements, please refer to the [System Requirements Specification (SRS)](REQUIREMENTS.md).
