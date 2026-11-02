FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_cache \
    KMP_DUPLICATE_LIB_OK=TRUE

WORKDIR /app

# build-essential: fallback compiler if a wheel is missing for this CPU; also provides libgomp (XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch index: avoids multi-GB CUDA wheels on Intel (amd64) Macs
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Bake all model/corpus downloads into the image so the container starts offline
RUN python -m spacy download en_core_web_sm \
    && python -m nltk.downloader -d /usr/local/share/nltk_data stopwords punkt averaged_perceptron_tagger \
    && python -c "from transformers import GPT2LMHeadModel, GPT2TokenizerFast; GPT2TokenizerFast.from_pretrained('distilgpt2'); GPT2LMHeadModel.from_pretrained('distilgpt2')"

COPY . .
RUN mkdir -p uploads

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=120s --retries=12 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
