# Project Viva & Research Q&A

This document provides detailed answers to critical research questions regarding the **Research-Grade AI Resume Detector**. These answers are based on the project's architectural implementation.

---

## 1️⃣ Input Data & Sources
**Q: What is the format and source of the data?**

*   **Formats**: The system accepts **Unstructured Data** in the following formats:
    *   **PDF** (Portable Document Format)
    *   **DOCX** (Microsoft Word Document)
    *   *Note: TXT/CSV are supported in the logic but the UI focuses on PDF/DOCX as they are standard for resumes.*
*   **Data Sources**:
    *   **User Uploads**: Real-time resumes uploaded by users via the Frontend.
    *   **Dataset (For Training/Testing)**: The system is designed to consume the **Kaggle Resume Dataset** (Human) and generated synthetic resumes (AI-generated using GPT-4/Claude) for validation.

---

## 2️⃣ Data Labeling Strategy
**Q: Is the data labeled or not?**

*   **Status**: The system deals with **Unlabeled Data** during the inference phase (when a user uploads a resume).
*   **Training Phase**: We utilize a **Weakly Supervised Approach**.
    *   Instead of manually hand-labeling thousands of resumes as "AI" or "Human" (which is expensive), we use programmatic rules to generate "Weak Labels".
    *   **AI-Generated Data**: We distinguish between completely **AI-Generated** text (free flow) and **Template-Based** text (structured forms). Our system targets the *content generation* style of AI, irrespective of the template.

---

## 3️⃣ Weakly Supervised Module
**Q: How does the module model and label data?**

*   **Concept**: Weak Supervision allows us to use noisy, heuristic-based rules to label data cheaply.
*   **Labeling Functions (Heuristics)**: We employ a set of "Labeling Functions" (LFs) in `services/weak_supervision.py`:
    1.  **Keyword Matching**: Checks for overused AI phrases (e.g., "delve into", "tapestry of", "meticulously crafted").
    2.  **Pattern Detection**: Identifies rigid, unnatural structural patterns common in AI outputs.
*   **Model**: These functions output a "Vote" (AI, Human, or Abstain). These votes are aggregated to form a "Weak Label" which acts as a proxy for ground truth.

---

## 4️⃣ Feature Extraction
**Q: What syntactic and semantic methods are used?**

We convert raw text into a numerical feature vector using two approaches:

### A. Syntactic Features (Structure-Based)
*   **Method**: **Stylometry** (using `textstat` and `spacy`).
*   **Features Extracted**:
    *   **Sentence Length Standard Deviation**: Humans write with varied rhythm (short and long sentences). AI is robotic and uniform.
    *   **Lexical Diversity (Type-Token Ratio)**: Measures vocabulary richness. AI tends to repeat words more than skilled human writers.

### B. Semantic Features (Meaning-Based)
*   **Method**: **Neural Perplexity** (using `DistilGPT2` Transformer).
*   **Explanation**: We use a pre-trained Language Model (LLM) to calculate "Perplexity" (Surprise).
    *   **Low Perplexity**: The model is "not surprised" by the text $\rightarrow$ Likely **AI-Generated** (predictable).
    *   **High Perplexity**: The model is "surprised" $\rightarrow$ Likely **Human-Written** (creative/chaotic).

---

## 5️⃣ Hybrid (ML + DL) Architecture
**Q: Why Hybrid? What is the bridge?**

*   **Purpose**:
    *   **Deep Learning (DL)**: Used for **Feature Extraction**. The Transformer model (`DistilGPT2`) is excellent at understanding context and probability (Perplexity) but is heavy and a "black box."
    *   **Machine Learning (ML)**: Used for the **Final Decision**. A lightweight classifier (Logistic Regression or Rule-Based Logic) takes the inputs from the DL model and the Syntactic features to make a discernible, explainable decision.
*   **The Bridge**:
    *   The **Feature Vector** acts as the bridge.
    *   *Process*: Raw Text $\xrightarrow{DL}$ Perplexity Score + Raw Text $\xrightarrow{NLP}$ Stylometric Scores = **Combined Feature Vector** $\xrightarrow{ML}$ Final Classification.
*   **Benefit**: This gives us the **Power of LLMs** (DL) with the **Speed and Explainability** of classical ML.
