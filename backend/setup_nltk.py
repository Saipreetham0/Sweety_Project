# NEW FILE
"""
Run once to download required NLTK corpora:
    python setup_nltk.py
"""
import nltk  # type: ignore[import]

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
print("NLTK data downloaded successfully.")
