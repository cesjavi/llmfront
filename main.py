"""
LLMFront - Backend FastAPI

Modos de operación (LLMFRONT_MODE en .env):
  - "api"   → Solo cloud: HF Inference API + Groq. Sin torch/transformers. ~100 MB RAM.
  - "local" → Full local: descarga modelos y corre con transformers + torch. RAM alta.
  - "both"  → Híbrido: habilita inferencia local Y cloud al mismo tiempo.

Variables clave en .env:
  LLMFRONT_MODE=api            # "api", "local" o "both" (default: "local")
  GROQ_AI_SEARCH_ENABLED=true  # habilita/deshabilita búsqueda IA en HF Hub
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Desactivar TensorFlow antes de cualquier import de transformers
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from config import APP_VERSION, LLMFRONT_MODE, LOCAL_MODE
from state import (
    HF_TOKEN, GROQ_API_KEY, TOGETHER_API_KEY, DEEPINFRA_API_KEY,
    FIREWORKS_API_KEY, BASETEN_API_KEY, NVIDIA_API_KEY,
)
import rag

# ─── Routers ──────────────────────────────────────────────────────────────────
from routers.models import router as models_router
from routers.chat import router as chat_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="LLMFront", description="HuggingFace LLM Chat Frontend", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    if request.url.path != "/system/info":
        logger.info(f"Request: {request.method} {request.url.path}")
    return await call_next(request)

# ─── Routers globales ─────────────────────────────────────────────────────────
app.include_router(models_router)
app.include_router(chat_router)

# ─── Routers solo en LOCAL_MODE ───────────────────────────────────────────────
if LOCAL_MODE:
    from routers.local_models import router as local_models_router
    from routers.compat import router as compat_router
    app.include_router(local_models_router)
    app.include_router(compat_router)

# ─── Endpoints básicos ────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return RedirectResponse(url="/app", status_code=307)

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "app": "LLMFront", "ollama_compat": True}

@app.get("/health")
async def health():
    return {"status": "ok", "version": APP_VERSION, "mode": LLMFRONT_MODE, "local_support": LOCAL_MODE}

@app.get("/config")
async def get_config():
    from config import GROQ_AI_SEARCH_ENABLED
    return {
        "mode": LLMFRONT_MODE,
        "local_mode": LOCAL_MODE,
        "groq_ai_search_enabled": GROQ_AI_SEARCH_ENABLED and bool(GROQ_API_KEY),
        "groq_available": bool(GROQ_API_KEY),
        "hf_available": bool(HF_TOKEN),
        "openrouter_available": bool(os.getenv("OPENROUTER_API_KEY", "")),
        "together_available": bool(TOGETHER_API_KEY),
        "deepinfra_available": bool(DEEPINFRA_API_KEY),
        "fireworks_available": bool(FIREWORKS_API_KEY),
        "baseten_available": bool(BASETEN_API_KEY),
        "nvidia_available": bool(NVIDIA_API_KEY),
    }

@app.get("/system/info")
async def system_info():
    from local.helpers import _system_info
    return _system_info()

# ─── Frontend estático ────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/app", response_class=HTMLResponse)
async def serve_frontend():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# ─── Entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading
    import time

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))

    def open_browser():
        time.sleep(1.5)
        url_host = "127.0.0.1" if host == "0.0.0.0" else host
        webbrowser.open(f"http://{url_host}:{port}/app")

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("main:app", host=host, port=port, reload=True, reload_excludes=["models_cache", "venv"])
