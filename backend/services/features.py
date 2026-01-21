import textstat
import spacy
import numpy as np
from collections import Counter
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Load Spacy Model (Ensure 'en_core_web_sm' is downloaded)
try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None

# Load Perplexity Model (DistilGPT2 for speed)
try:
    _tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
    _model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    _model.eval()
except Exception as e:
    print(f"Warning: Could not load GPT-2 model: {e}")
    _tokenizer = None
    _model = None

class FeatureExtractor:
    @staticmethod
    def extract_features(text: str) -> dict:
        features = {}
        
        # 1. Stylometric Features (Linguistic Style)
        # ------------------------------------------
        features["flesch_reading_ease"] = textstat.flesch_reading_ease(text)
        features["reading_time"] = textstat.reading_time(text)
        features["lexical_diversity"] = FeatureExtractor._lexical_diversity(text)
        features["perplexity"] = FeatureExtractor._calculate_perplexity(text)

        # Spacy Analysis
        if nlp:
            doc = nlp(text)
            
            # POS Tag Distribution (Adjectives/Verbs)
            pos_counts = Counter([token.pos_ for token in doc])
            total_tokens = len(doc)
            features["adj_ratio"] = pos_counts.get("ADJ", 0) / (total_tokens + 1e-6)
            features["verb_ratio"] = pos_counts.get("VERB", 0) / (total_tokens + 1e-6)
            
            # Sentence Length Std Dev (AI is often uniform)
            sent_lens = [len(sent) for sent in doc.sents]
            features["sent_len_std"] = np.std(sent_lens) if sent_lens else 0
            features["sent_len_mean"] = np.mean(sent_lens) if sent_lens else 0

        # 2. Structural Features
        # ----------------------
        features["bullet_density"] = text.count("•") + text.count("-") / (len(text.splitlines()) + 1)
        
        return features

    @staticmethod
    def _calculate_perplexity(text: str) -> float:
        if not _model or not _tokenizer or not text.strip():
            return 0.0
            
        encodings = _tokenizer(text, return_tensors="pt")
        max_length = _model.config.n_positions
        stride = 512
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = _model(input_ids, labels=target_ids)
                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        if not nlls:
             return 0.0

        ppl = torch.exp(torch.stack(nlls).mean())
        return float(ppl)

    @staticmethod
    def _lexical_diversity(text: str) -> float:
        tokens = text.lower().split()
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)
