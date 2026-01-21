# AI-Powered Resume Detector

A robust full-stack application designed to detect AI-generated resumes using a Weakly Supervised Hybrid Learning approach. The system analyzes uploaded resumes (PDF, DOCX) to identify linguistic patterns typical of AI models.

## 🌟 Features
- **Files Supported**: PDF, DOCX, TXT.
- **Analysis Pipeline**:
    - **Text Extraction**: Robust parsing of document formats.
    - **Heuristic Detection**: Identifies "AI-isms" (e.g., "delve into", "tapestry").
    - **Hybrid Scoring**: Combines weak supervision signals into a confidence score.
- **Modern UI**: Clean, responsive dashboard built with Shadcn UI.
- **FastAPI Backend**: High-performance async Python backend.

## 🏗 Project Architecture

```
Sweety_Project/
├── backend/                 # FastAPI Application
│   ├── routers/             # API Endpoints (Upload, Analyze)
│   ├── services/            # Core Logic (Extraction, ML, Preprocessing)
│   ├── data/                # Data storage (Uploaded files, synthetic data)
│   ├── requirements.txt     # Python Dependencies
│   └── main.py              # Entry Point
├── frontend/                # Next.js 16 Application
│   ├── app/                 # App Router (Pages)
│   └── components/          # Shadcn UI Components
├── README.md                # Documentation
├── start_app.sh             # One-click startup script
└── verify_setup.py          # Backend verification script
```

## � Quick Start (Recommended)

Run the entire application with a single command:

```bash
chmod +x start_app.sh
./start_app.sh
```
*This starts the Backend on http://localhost:8000 and Frontend on http://localhost:3000.*

---

## 🛠 Manual Installation

### 1. Backend Setup
**Prerequisite**: Python 3.10+

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### 2. Frontend Setup
**Prerequisite**: Node.js 18+

```bash
cd frontend
npm install
npm run dev
```

## 📡 API Documentation

### `POST /upload/`
Uploads a resume file to the server.
- **Body**: `file` (Multipart/Form-Data)
- **Response**: `{"filename": "resume.pdf", "status": "uploaded"}`

### `POST /analyze/{filename}`
Triggers analysis for a previously uploaded file.
- **Path Param**: `filename`
- **Response**:
  ```json
  {
    "is_ai_generated": true,
    "confidence": 0.85,
    "explanation": ["Contains common AI phrase: 'delve into'"],
    "raw_heuristics": {...}
  }
  ```

## 🧪 Verification
You can verify the backend logic without running the server by executing the test script:
```bash
python3 verify_setup.py
```

## � Future Improvements
- [ ] **Database Integration**: Store analysis history in Supabase.
- [ ] **Model Training**: Replace the heuristic prototype with a fine-tuned DistilBERT model.
- [ ] **Authentication**: Add user login for saving reports.
