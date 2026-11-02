import os
import sys
import glob
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.extraction import TextExtractor
from services.preprocessing import TextPreprocessor
from services.weak_supervision import WeakSupervision
from services.model import ResumeModel

model = ResumeModel()
ws = WeakSupervision()

candidates = [
    # 10 AI Benchmarks
    ("AI_Candidate_1.txt", "AI Benchmark 1"),
    ("AI_Candidate_2.txt", "AI Benchmark 2"),
    ("AI_Candidate_3.txt", "AI Benchmark 3"),
    ("AI_Candidate_4.txt", "AI Benchmark 4"),
    ("AI_Candidate_5.txt", "AI Benchmark 5"),
    ("AI_Candidate_6.txt", "AI Benchmark 6"),
    ("AI_Candidate_7.txt", "AI Benchmark 7"),
    ("AI_Candidate_8.txt", "AI Benchmark 8"),
    ("AI_Candidate_9.txt", "AI Benchmark 9"),
    ("AI_Candidate_10.txt", "AI Benchmark 10"),
    # 10 Human Benchmarks
    ("Human_Candidate_1.txt", "Human Benchmark 1"),
    ("Human_Candidate_2.txt", "Human Benchmark 2"),
    ("Human_Candidate_3.txt", "Human Benchmark 3"),
    ("Human_Candidate_4.txt", "Human Benchmark 4"),
    ("Human_Candidate_5.txt", "Human Benchmark 5"),
    ("Human_Candidate_6.txt", "Human Benchmark 6"),
    ("Human_Candidate_7.txt", "Human Benchmark 7"),
    ("Human_Candidate_8.txt", "Human Benchmark 8"),
    ("Human_Candidate_9.txt", "Human Benchmark 9"),
    ("Human_Candidate_10.txt", "Human Benchmark 10"),
    # 6 Real Uploaded Candidate Resumes
    ("Sai Preetham Koyyala Resume (1).pdf", "Real Resume: Sai Preetham"),
    ("Farheen DE CV Amazon.pdf", "Real Resume: Farheen DE CV"),
    ("Chinna  Resume .pdf", "Real Resume: Chinna"),
    ("KSP RESUME.pdf", "Real Resume: KSP"),
    ("new (5).pdf", "Real Resume: new (5)"),
    ("Action plan.docx", "Real Doc: Action plan"),
]

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
search_dirs = [
    os.path.join(backend_dir, "uploads"),
    os.path.join(backend_dir, "data", "ai"),
    os.path.join(backend_dir, "data", "human"),
]

results = []

for filename, desc in candidates:
    file_path = None
    for d in search_dirs:
        # direct name
        cand = os.path.join(d, filename)
        if os.path.isfile(cand):
            file_path = cand
            break
        # search with glob
        for f in glob.glob(os.path.join(d, "*")):
            if f.endswith(filename) or filename in os.path.basename(f):
                file_path = f
                break
        if file_path:
            break

    if not file_path or not os.path.isfile(file_path):
        results.append({"name": desc, "status": f"File Not Found: {filename}"})
        continue

    try:
        text = TextExtractor.extract_from_file(file_path)
        clean = TextPreprocessor.clean_text(text)
        heuristics = ws.apply_heuristics(clean)
        res = model.predict(clean, heuristics, skip_perplexity=True)

        m_results = res.get("model_results", {})
        rf = m_results.get("random_forest", {})
        xgb = m_results.get("xgboost", {})
        ens = m_results.get("ensemble", {})

        results.append({
            "name": desc,
            "filename": os.path.basename(file_path),
            "status": "OK",
            "label": res.get("label", "unknown"),
            "confidence": res.get("confidence", 0.0),
            "rf": rf,
            "xgb": xgb,
            "ens": ens,
            "ai_phrases": len(heuristics.get("matched_ai_phrases", [])),
            "tmpl_patterns": len(heuristics.get("matched_template_patterns", [])),
        })
    except Exception as e:
        results.append({
            "name": desc,
            "filename": filename,
            "status": f"Error: {e}"
        })

hdr_num = "#"
hdr_name = "Resume / Document"
hdr_ens = "Ensemble"
hdr_conf = "Conf"
hdr_rf = "Random Forest"
hdr_xgb = "XGBoost"
hdr_signs = "AI / Tmpl Signs"

print(f"\n==================================== TEST RESULTS: {len(results)} RESUMES ====================================")
print(f"{hdr_num:<3} | {hdr_name:<28} | {hdr_ens:<15} | {hdr_conf:<6} | {hdr_rf:<15} | {hdr_xgb:<15} | {hdr_signs}")
print("-" * 115)

for i, r in enumerate(results, 1):
    if r["status"] != "OK":
        print(f"{i:<3} | {r['name'][:28]:<28} | {r['status']}")
        continue
    ens_label = r["label"]
    ens_conf = f"{r['confidence']*100:.1f}%"
    rf_label = r["rf"].get("label", "N/A")
    xgb_label = r["xgb"].get("label", "N/A")
    signs = f"AI: {r['ai_phrases']}, Tmpl: {r['tmpl_patterns']}"

    print(f"{i:<3} | {r['name'][:28]:<28} | {ens_label:<15} | {ens_conf:<6} | {rf_label:<15} | {xgb_label:<15} | {signs}")

print("=" * 115)
