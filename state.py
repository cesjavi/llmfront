import threading
import os
from typing import Dict

# ─── Estado Global ────────────────────────────────────────────────────────────
_model_lock = threading.Lock()
_download_lock = threading.Lock()

_loaded_models: Dict[str, dict] = {}
_download_state: Dict[str, dict] = {}

HF_TOKEN = os.getenv("HF_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
BASETEN_API_KEY = os.getenv("BASETEN_API_KEY", "")
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY", "")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(os.getcwd(), "models_cache"))
os.makedirs(MODELS_DIR, exist_ok=True)

# _is_downloaded y _is_partial viven en local/helpers.py
# No importar desde aquí — usar: from local.helpers import _is_downloaded, _is_partial
