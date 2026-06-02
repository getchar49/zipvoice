import os

RESULTS_DIR      = os.getenv("RESULTS_DIR", "results")
VOICES_DIR       = os.getenv("VOICES_DIR", "voices")

# Model settings
MODEL_NAME       = os.getenv("MODEL_NAME", "zipvoice")  
ZIPVOICE_MODEL_DIR = os.getenv("ZIPVOICE_MODEL_DIR", "checkpoint")  # set to fully offline dir (model.pt, model.json, tokens.txt)
VOCOS_LOCAL_DIR  = os.getenv("VOCOS_LOCAL_DIR", None)      # set for offline vocoder (config.yaml, pytorch_model.bin)

# Inference
DEVICE           = os.getenv("DEVICE", "cuda")
TOKENIZER        = os.getenv("TOKENIZER", "espeak")
LANG_TOKENIZER   = os.getenv("LANG_TOKENIZER", "vi")
MAX_DURATION     = float(os.getenv("MAX_DURATION", "100"))  # per internal batch cap (sec)
MAX_CONCURRENT   = int(os.getenv("MAX_CONCURRENT", "5"))
USE_MULTIPLE_MODELS=True

# LLM Normalizer
LLM_API_URL      = os.getenv("LLM_API_URL", "http://x.x.x.x:x/v1/chat/completions")
LLM_MODEL        = os.getenv("LLM_MODEL", "llm-model")
LLM_API_KEY      = os.getenv("LLM_API_KEY", "dummy")
LLM_TIMEOUT      = int(os.getenv("LLM_TIMEOUT", "10"))         # seconds per request
USE_LLM_NORMALIZER = os.getenv("USE_LLM_NORMALIZER", "true").lower() == "true"

# Bracket inference params (for <X> letter segments)
BRACKET_SPEED    = float(os.getenv("BRACKET_SPEED", "0.4"))
BRACKET_NUM_STEP = int(os.getenv("BRACKET_NUM_STEP", "32"))

# Double punctuation experiment (. → .. and , → ,,)
USE_DOUBLE_PUNCTUATION = os.getenv("USE_DOUBLE_PUNCTUATION", "false").lower() == "true"