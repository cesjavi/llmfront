"""
config.py – Configuración global de la aplicación.
Importar desde aquí para evitar repetición en cada módulo.
"""
import os
from dotenv import load_dotenv

load_dotenv()

APP_VERSION = "0.9"
LLMFRONT_MODE = os.getenv("LLMFRONT_MODE", "local").lower()
LOCAL_MODE = LLMFRONT_MODE in ("local", "both")
GROQ_AI_SEARCH_ENABLED = os.getenv("GROQ_AI_SEARCH_ENABLED", "true").lower() == "true"

# Proveedores cloud soportados
CLOUD_PROVIDERS = {"groq", "openrouter", "together", "deepinfra", "fireworks", "baseten"}

# URLs base de cada proveedor OpenAI-compatible
PROVIDER_BASE_URLS = {
    "groq":       "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "together":   "https://api.together.xyz/v1",
    "deepinfra":  "https://api.deepinfra.com/v1/openai",
    "fireworks":  "https://api.fireworks.ai/inference/v1",
    "baseten":    "https://inference.baseten.co/v1",
}
