"""
Dataset Loader Module — Dataset A (Corpus) and Dataset B (Heuristics).

Loads from Excel spreadsheets if available:
  - Dataset_A_Combined_Resume_Corpus.xlsx
  - Dataset_B_Keyword_Heuristic_Signals.xlsx
Falls back to robust embedded scientific vocabularies and local candidate corpora.
"""

from __future__ import annotations

import os
from typing import Dict, List, Any

import pandas as pd  # type: ignore[import]

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT_DIR = os.path.dirname(_BACKEND_DIR)

_DATASET_B_PATHS = [
    os.path.join(_BACKEND_DIR, "data", "Dataset_B_Keyword_Heuristic_Signals.xlsx"),
    os.path.join(_ROOT_DIR, "Dataset_B_Keyword_Heuristic_Signals.xlsx"),
]

_DATASET_A_PATHS = [
    os.path.join(_BACKEND_DIR, "data", "Dataset_A_Combined_Resume_Corpus.xlsx"),
    os.path.join(_ROOT_DIR, "Dataset_A_Combined_Resume_Corpus.xlsx"),
]

# ── 50 Canonical AI Signature Phrases (Dataset B) ───────────────────────────
_DEFAULT_AI_KEYWORDS: List[Dict[str, str]] = [
    {"phrase": "delve into", "signal_strength": "very high"},
    {"phrase": "tapestry of", "signal_strength": "very high"},
    {"phrase": "meticulously crafted", "signal_strength": "very high"},
    {"phrase": "testament to", "signal_strength": "very high"},
    {"phrase": "leverage the power of", "signal_strength": "very high"},
    {"phrase": "poised to", "signal_strength": "high"},
    {"phrase": "spearheading cross-functional", "signal_strength": "high"},
    {"phrase": "foster a culture of", "signal_strength": "high"},
    {"phrase": "synergistic growth", "signal_strength": "high"},
    {"phrase": "unwavering commitment", "signal_strength": "high"},
    {"phrase": "dynamic and detail-oriented", "signal_strength": "high"},
    {"phrase": "proven track record", "signal_strength": "high"},
    {"phrase": "navigated the complexities", "signal_strength": "high"},
    {"phrase": "cutting-edge technologies", "signal_strength": "high"},
    {"phrase": "game-changing solutions", "signal_strength": "high"},
    {"phrase": "beacon of innovation", "signal_strength": "high"},
    {"phrase": "seamlessly integrate", "signal_strength": "high"},
    {"phrase": "plethora of", "signal_strength": "medium"},
    {"phrase": "in the realm of", "signal_strength": "medium"},
    {"phrase": "catalyst for change", "signal_strength": "medium"},
    {"phrase": "pivotal role in", "signal_strength": "medium"},
    {"phrase": "holistic approach", "signal_strength": "medium"},
    {"phrase": "rich tapestry", "signal_strength": "very high"},
    {"phrase": "fast-paced landscape", "signal_strength": "high"},
    {"phrase": "harnessing the power", "signal_strength": "high"},
    {"phrase": "results-driven professional with a proven track record", "signal_strength": "very high"},
    {"phrase": "passionate advocate for", "signal_strength": "medium"},
    {"phrase": "demonstrated acumen in", "signal_strength": "medium"},
    {"phrase": "orchestrated comprehensive", "signal_strength": "high"},
    {"phrase": "drive impactful outcomes", "signal_strength": "high"},
    {"phrase": "leveraging cutting-edge", "signal_strength": "high"},
    {"phrase": "underscoring a commitment", "signal_strength": "high"},
    {"phrase": "steer organizational success", "signal_strength": "medium"},
    {"phrase": "strategic foresight", "signal_strength": "medium"},
    {"phrase": "spearheaded the design and implementation", "signal_strength": "high"},
    {"phrase": "driving paradigm shifts", "signal_strength": "very high"},
    {"phrase": "reimagining possibilities", "signal_strength": "high"},
    {"phrase": "empowering stakeholders", "signal_strength": "medium"},
    {"phrase": "delivering bespoke solutions", "signal_strength": "high"},
    {"phrase": "fostering collaborative ecosystems", "signal_strength": "high"},
    {"phrase": "transformative insights", "signal_strength": "medium"},
    {"phrase": "synergize cross-functional teams", "signal_strength": "high"},
    {"phrase": "accelerating digital transformation", "signal_strength": "high"},
    {"phrase": "elevating operational efficiency", "signal_strength": "medium"},
    {"phrase": "groundbreaking methodologies", "signal_strength": "medium"},
    {"phrase": "unparalleled dedication", "signal_strength": "medium"},
    {"phrase": "holistically addressing", "signal_strength": "medium"},
    {"phrase": "pioneering novel strategies", "signal_strength": "medium"},
    {"phrase": "exemplary standard of quality", "signal_strength": "high"},
    {"phrase": "cultivating high-performance environments", "signal_strength": "high"},
]

# ── 24 Template Pattern Keywords & Placeholders (Dataset B) ─────────────────
_DEFAULT_TEMPLATE_PATTERNS: List[Dict[str, str]] = [
    {"phrase": "[company name]", "signal_strength": "very high"},
    {"phrase": "[position title]", "signal_strength": "very high"},
    {"phrase": "[date range]", "signal_strength": "very high"},
    {"phrase": "[your name]", "signal_strength": "very high"},
    {"phrase": "[job title]", "signal_strength": "very high"},
    {"phrase": "[university name]", "signal_strength": "very high"},
    {"phrase": "[phone number]", "signal_strength": "high"},
    {"phrase": "[email address]", "signal_strength": "high"},
    {"phrase": "[city, state]", "signal_strength": "high"},
    {"phrase": "references available upon request", "signal_strength": "high"},
    {"phrase": "i hereby declare that all information", "signal_strength": "very high"},
    {"phrase": "curriculum vitae", "signal_strength": "medium"},
    {"phrase": "seeking a challenging position in a reputed organization", "signal_strength": "very high"},
    {"phrase": "career objective:", "signal_strength": "medium"},
    {"phrase": "declare that the above mentioned information is true", "signal_strength": "very high"},
    {"phrase": "lorem ipsum dolor sit amet", "signal_strength": "very high"},
    {"phrase": "[insert skill]", "signal_strength": "very high"},
    {"phrase": "[bullet point]", "signal_strength": "very high"},
    {"phrase": "[achievement description]", "signal_strength": "very high"},
    {"phrase": "place: [city]", "signal_strength": "high"},
    {"phrase": "date: [dd/mm/yyyy]", "signal_strength": "very high"},
    {"phrase": "signature: [your signature]", "signal_strength": "very high"},
    {"phrase": "to secure a challenging position where i can effectively contribute", "signal_strength": "high"},
    {"phrase": "seeking an entry-level position to utilize my skills", "signal_strength": "medium"},
]


def load_ai_keywords() -> List[Dict[str, str]]:
    """Loads 50 AI phrases from Excel or default embedded vocabulary."""
    for path in _DATASET_B_PATHS:
        if os.path.exists(path):
            try:
                df = pd.read_excel(path, sheet_name=0)
                cols = {str(c).lower().strip(): c for c in df.columns}
                phrase_col = cols.get("phrase") or cols.get("keyword") or cols.get("ai_phrase")
                strength_col = cols.get("signal_strength") or cols.get("strength") or cols.get("weight")
                if phrase_col:
                    results = []
                    for _, row in df.iterrows():
                        p = str(row.get(phrase_col, "")).strip()
                        if p and p.lower() != "nan":
                            s = str(row.get(strength_col, "medium")).strip() if strength_col else "medium"
                            results.append({"phrase": p, "signal_strength": s})
                    if len(results) >= 10:
                        return results
            except Exception:
                pass
    return list(_DEFAULT_AI_KEYWORDS)


def load_template_patterns() -> List[Dict[str, str]]:
    """Loads 24 template patterns from Excel or default embedded vocabulary."""
    for path in _DATASET_B_PATHS:
        if os.path.exists(path):
            try:
                xls = pd.ExcelFile(path)
                target_sheet = None
                for s in xls.sheet_names:
                    if "template" in s.lower() or "pattern" in s.lower():
                        target_sheet = s
                        break
                if target_sheet:
                    df = pd.read_excel(path, sheet_name=target_sheet)
                    cols = {str(c).lower().strip(): c for c in df.columns}
                    phrase_col = cols.get("phrase") or cols.get("pattern") or cols.get("keyword")
                    strength_col = cols.get("signal_strength") or cols.get("strength")
                    if phrase_col:
                        results = []
                        for _, row in df.iterrows():
                            p = str(row.get(phrase_col, "")).strip()
                            if p and p.lower() != "nan":
                                s = str(row.get(strength_col, "medium")).strip() if strength_col else "medium"
                                results.append({"phrase": p, "signal_strength": s})
                        if len(results) >= 5:
                            return results
            except Exception:
                pass
    return list(_DEFAULT_TEMPLATE_PATTERNS)


def load_training_data() -> pd.DataFrame:
    """
    Loads training dataset A from Excel, or the labelled corpus in backend/data/{human,ai},
    combined with synthetic templates.
    """
    for path in _DATASET_A_PATHS:
        if os.path.exists(path):
            try:
                df = pd.read_excel(path)
                cols = {str(c).lower().strip(): c for c in df.columns}
                text_col = cols.get("resume_text") or cols.get("text") or cols.get("content")
                label_col = cols.get("label") or cols.get("category") or cols.get("class")
                if text_col and label_col:
                    records = []
                    for idx, row in df.iterrows():
                        t = str(row.get(text_col, "")).strip()
                        lbl = str(row.get(label_col, "human_written")).strip().lower()
                        if t and t.lower() != "nan":
                            records.append({
                                "resume_text": t,
                                "label": lbl,
                                "resume_id": f"EXCEL_{idx}",
                                "confidence_score": 1.0,
                            })
                    if len(records) >= 10:
                        return pd.DataFrame(records)
            except Exception:
                pass

    # Labelled corpus: label comes from the folder, never from user uploads.
    records: List[Dict[str, object]] = []
    for label, d in (
        ("human_written", os.path.join(_BACKEND_DIR, "data", "human")),
        ("ai_generated",  os.path.join(_BACKEND_DIR, "data", "ai")),
    ):
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".txt"):
                continue
            with open(os.path.join(d, fname), "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
            if content:
                records.append({
                    "resume_text": content,
                    "label": label,
                    "resume_id": fname,
                    "confidence_score": 0.95,
                })

    # Add realistic human technical resumes to ensure model handles technical domain vocabulary
    realistic_human_samples = [
        (
            "Alex Morgan\nFull Stack & IoT Developer | alex.morgan@example.com\n"
            "Skills: JavaScript, C++, Embedded C, Python, React, Node.js, Express, AWS IoT, Docker, MQTT, Tailwind CSS\n\n"
            "Experience:\n"
            "• Designed and deployed scalable IoT telemetry pipeline processing 50k sensor messages daily.\n"
            "• Implemented full-stack dashboard in React and Tailwind connected to Node.js backend.\n"
            "• Debugged firmware memory leaks on ESP32 microcontrollers using PlatformIO.\n"
            "• Built automated CI/CD pipeline using GitHub Actions to deploy containerized microservices.\n\n"
            "Education:\nB.Tech in Computer Science and Engineering."
        ),
        (
            "Data Engineer | AWS & Python Specialist\n"
            "Skills: Python, PySpark, Snowflake, dbt, SQL, PostgreSQL, Apache Airflow, Docker, Kubernetes\n\n"
            "Work Experience:\n"
            "• Developed scalable ETL pipelines transforming 2TB of daily transaction records in Snowflake.\n"
            "• Optimized slow SQL analytical queries reducing dashboard load times by 40%.\n"
            "• Integrated Apache Kafka streaming topic for real-time fraud monitoring.\n"
            "• Wrote unit and integration tests achieving 85% code coverage across data workflows.\n\n"
            "Education:\nBachelor of Technology in Information Technology."
        ),
        (
            "Frontend Engineer\n"
            "Summary: 4 years of experience building accessible and responsive user interfaces.\n"
            "Tech Stack: TypeScript, React, Next.js, Redux Toolkit, HTML5, CSS3, Jest, Webpack\n\n"
            "Key Projects:\n"
            "• Redesigned customer checkout portal improving mobile conversion rates by 18%.\n"
            "• Created reusable component library compliant with WCAG 2.1 AA accessibility guidelines.\n"
            "• Reduced bundle size by 32% by implementing code-splitting and dynamic imports.\n"
            "• Collaborated with UX designers and product managers in 2-week agile sprints."
        ),
        (
            "Backend Systems Developer\n"
            "Experience:\n"
            "• Built high-throughput gRPC services in Go handling 12,000 requests per second.\n"
            "• Migrated legacy monolithic Java service to distributed microservices on AWS EKS.\n"
            "• Configured Redis cluster for caching frequently accessed user authentication sessions.\n"
            "• Diagnosed database deadlocks and tuned connection pool configurations in PostgreSQL."
        ),
    ]

    for i, t in enumerate(realistic_human_samples):
        records.append({
            "resume_text": t,
            "label": "human_written",
            "resume_id": f"HUMAN_TECH_{i+1}",
            "confidence_score": 1.0,
        })

    # Add template-based samples to ensure 3-class balance
    template_samples = [
        (
            "Name: [Your Name]\nContact: [Email Address] | [Phone Number] | [City, State]\n\n"
            "Career Objective:\nSeeking a challenging position in a reputed organization where I can contribute my skills.\n\n"
            "Education:\n[Degree Name] - [University Name] ([Date Range])\n\n"
            "Experience:\n[Job Title] | [Company Name] | [Date Range]\n"
            "- [Bullet point describing key responsibilities]\n"
            "- [Bullet point describing projects]\n\n"
            "Declaration:\nI hereby declare that all information provided above is true to the best of my knowledge.\n"
            "References available upon request."
        ),
        (
            "CURRICULUM VITAE\n[Your Name] - [Job Title]\n\n"
            "Objective: To secure a challenging position where I can effectively contribute my knowledge.\n\n"
            "Key Skills:\n- [Insert Skill 1]\n- [Insert Skill 2]\n- [Insert Skill 3]\n\n"
            "Professional Experience:\n[Company Name] | [Position Title] | [Date Range]\n"
            "- [Achievement description]\n"
            "References available upon request."
        ),
        (
            "RESUME\n[Your Name]\nAddress: [City, State]\n\n"
            "Career Objective: Seeking an entry-level position to utilize my skills for mutual organizational growth.\n\n"
            "Academic Record:\n[Degree] from [University Name], Graduated in [Date Range]\n\n"
            "Declaration:\nI declare that the above mentioned information is true to my belief.\n"
            "Date: [DD/MM/YYYY]\nSignature: [Your Signature]"
        ),
        (
            "Candidate Profile: [Your Name]\nTarget Role: [Position Title]\n\n"
            "Summary:\n[Company Name] references available upon request. Seeking a challenging position in a reputed organization.\n\n"
            "Experience:\n- [Job Title] at [Company Name] from [Date Range].\n"
            "I hereby declare that all information provided above is true to the best of my knowledge."
        ),
    ]

    for i, t in enumerate(template_samples):
        records.append({
            "resume_text": t,
            "label": "template_based",
            "resume_id": f"TMPL_SAMPLE_{i+1}",
            "confidence_score": 0.95,
        })

    return pd.DataFrame(records)
