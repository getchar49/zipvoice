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