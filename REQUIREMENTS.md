# Software Requirements Specification (SRS)
## Project Title: Research-Grade AI Resume Detector

### 1. Introduction
This project aims to develop a scientifically rigorous system to detect AI-generated resumes using a Weakly Supervised Hybrid Learning approach. The system analyzes resumes to distinguish between human-written and AI-generated content (e.g., from ChatGPT, Claude).

### 2. Implementation Environment

#### 2.1 Hardware Requirements (Minimum)
-   **Processor**: Intel Core i5 (8th Gen) or equivalent / Apple Silicon M1
-   **RAM**: 8 GB (16 GB Recommended for smoother ML inference)
-   **Storage**: 5 GB free disk space
-   **Internet**: Required for downloading pre-trained models (DistilGPT2, Spacy)

#### 2.2 Software Requirements
-   **Operating System**: Windows 10/11, macOS, or Linux
-   **Backend Language**: Python 3.10+
-   **Frontend Framework**: Node.js 18+ (Next.js 16)
-   **Containerization**: Docker & Docker Compose (Optional, for easy deployment)
-   **Browser**: Google Chrome, Firefox, or Safari

### 3. Functional Requirements

#### 3.1 User Interface (Frontend)
-   **File Upload**: Users must be able to upload resume files (PDF, DOCX).
-   **Dashboard**: A responsive dashboard to display analysis results.
-   **Visualization**: Real-time charts showing "Human vs. AI" probability.
-   **Detailed Explanations**: The system must provide reasons for its decision (e.g., "Low Perplexity", "Repetitive Sentence Structure").

#### 3.2 Analysis Engine (Backend)
-   **Text Extraction**: Automatically extract text from PDF and DOCX files.
-   **Text Preprocessing**: Clean text by removing special characters, excessive whitespace, and stop words.
-   **Feature Engineering**:
    -   Calculate **Perplexity** using GPT-2.
    -   Calculate **Burstiness** (Sentence length variation).
    -   Calculate **Lexical Diversity** (Type-Token Ratio).
-   **Prediction Model**: A hybrid model combining heuristic rules (Weak Supervision) and statistical features to classify the text.

### 4. Non-Functional Requirements
-   **Performance**: Analysis should complete within 10 seconds for a standard 2-page resume.
-   **Accuracy**: The system should achieve >85% accuracy on benchmark test sets.
-   **Reliability**: The system should handle malformed files gracefully without crashing.
-   **Scalability**: The modular architecture (Frontend separated from Backend) allows for future scalability.

### 5. System Dependencies
-   **Python Libraries**: `fastapi`, `torch`, `transformers`, `scikit-learn`, `spacy`, `textstat`, `python-docx`, `pypdf`.
-   **JavaScript Libraries**: `react`, `next`, `tailwindcss`, `lucide-react`.
