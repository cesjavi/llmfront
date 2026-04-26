"""
LLMFront - Backend FastAPI

Modos de operación (LLMFRONT_MODE en .env):
  - "api"   → Solo cloud: HF Inference API + Groq. Sin torch/transformers. ~100 MB RAM.
  - "local" → Full local: descarga modelos y corre con transformers + torch. RAM alta.
  - "both"  → Híbrido: habilita inferencia local Y cloud al mismo tiempo.

Variables clave en .env:
  LLMFRONT_MODE=api            # "api" o "local" (default: "local")
  GROQ_AI_SEARCH_ENABLED=true  # habilita/deshabilita búsqueda IA en HF Hub
"""

import os
import json
import re
import shutil
import asyncio
import logging
import base64
from pathlib import Path
from io import BytesIO
from threading import Thread, Lock
from typing import AsyncGenerator, Optional
from datetime import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI
import threading
import ctypes
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from huggingface_hub import InferenceClient, snapshot_download
import requests
import rag

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_VERSION = "0.9"

# ─── Configuración de modo ────────────────────────────────────────────────────
LLMFRONT_MODE = os.getenv("LLMFRONT_MODE", "local").lower()
LOCAL_MODE = LLMFRONT_MODE in ("local", "both")
GROQ_AI_SEARCH_ENABLED = os.getenv("GROQ_AI_SEARCH_ENABLED", "true").lower() in ("1", "true", "yes")

HF_TOKEN = os.getenv("HF_TOKEN", "")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY", "")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
BASETEN_API_KEY = os.getenv("BASETEN_API_KEY", "")
MODELS_CACHE_DIR = Path("./models_cache")
if LOCAL_MODE:
    MODELS_CACHE_DIR.mkdir(exist_ok=True)

_mode_label = (
    "BOTH 🔀  (local + cloud habilitados)" if LLMFRONT_MODE == "both"
    else "LOCAL 🖥️  (inferencia local habilitada)" if LOCAL_MODE
    else "API ☁️  (solo cloud, sin torch)"
)
logger.info(f"🚀 LLMFront modo: {_mode_label}")
if GROQ_AI_SEARCH_ENABLED and os.getenv("GROQ_API_KEY", ""):
    logger.info("🔍 Búsqueda IA (Groq): HABILITADA")
elif GROQ_AI_SEARCH_ENABLED:
    logger.info("⚠️  Búsqueda IA (Groq): habilitada pero sin GROQ_API_KEY")
else:
    logger.info("🔍 Búsqueda IA (Groq): DESHABILITADA")

app = FastAPI(title="LLMFront", description="HuggingFace LLM Chat Frontend", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Middleware para Debug (Ver qué pide Twinny) ─────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    path = request.url.path
    # No loggear /system/info para no ensuciar la consola
    if path != "/system/info":
        logger.info(f"🔔 Request: {request.method} {path}")
    response = await call_next(request)
    return response

@app.get("/healthz")
async def root_health():
    return {"status": "ok", "app": "LLMFront", "ollama_compat": True}

# download_state[model_key] = {status, progress, message, size_gb}
_download_state: dict[str, dict] = {}
_download_lock = Lock()

# Modelos cargados en memoria: model_key -> {pipeline, model_id, device, quantization}
_loaded_models: dict[str, dict] = {}
_model_lock = Lock()


# ─── Modelos Pydantic ─────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str

class ImageItem(BaseModel):
    base64: str
    mime_type: str

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    system_prompt: Optional[str] = "You are a helpful assistant."
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    stream: bool = True
    hf_token: Optional[str] = None
    use_local: bool = False          # True = usar modelo local descargado
    provider: str = "hf"             # hf, local, groq, openrouter
    api_key: Optional[str] = None
    images: Optional[list[ImageItem]] = None

class ModelSearchRequest(BaseModel):
    query: str = ""
    task: str = "text-generation"
    size_filter: str = "any"
    limit: int = 20
    hf_token: Optional[str] = None
    provider: str = "hf"
    api_key: Optional[str] = None
    use_ai_search: bool = False

class ModelApiCheckRequest(BaseModel):
    model_id: str
    provider: str = "hf"
    api_key: Optional[str] = None
    hf_token: Optional[str] = None
    
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


class DownloadRequest(BaseModel):
    model_id: str
    hf_token: Optional[str] = None
    quantization: str = "none"       # "none" | "4bit" | "8bit"

class OllamaMessage(BaseModel):
    role: str
    content: str

class OllamaChatRequest(BaseModel):
    model: str
    messages: list[OllamaMessage]
    options: Optional[dict] = {}
    stream: bool = True

class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    system: Optional[str] = None
    options: Optional[dict] = {}
    stream: bool = True

class LoadModelRequest(BaseModel):
    model_id: str
    quantization: str = "none"       # "none" | "4bit" | "8bit"
    device: str = "auto"             # "auto" | "cpu" | "cuda"


# ─── Modelos curados ──────────────────────────────────────────────────────────

FEATURED_MODELS = [
    {
        "id": "mistralai/Mistral-7B-Instruct-v0.3",
        "name": "Mistral 7B Instruct v0.3",
        "description": "Rápido y capaz. Ideal para uso general. ~14 GB.",
        "tags": ["chat", "instruct", "7B"],
        "likes": 12000,
        "size_gb": 14.5,
    },
    {
        "id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "name": "Llama 3 8B Instruct",
        "description": "Llama 3 de Meta, muy buena calidad de razonamiento. ~16 GB.",
        "tags": ["chat", "instruct", "8B", "llama3"],
        "likes": 25000,
        "size_gb": 16.0,
    },
    {
        "id": "google/gemma-2-9b-it",
        "name": "Gemma 2 9B IT",
        "description": "Modelo de Google con excelente rendimiento. ~18 GB.",
        "tags": ["chat", "instruct", "9B", "google"],
        "likes": 8000,
        "size_gb": 18.0,
    },
    {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "name": "Qwen 2.5 7B Instruct",
        "description": "Multilingüe y con soporte para código. ~15 GB.",
        "tags": ["chat", "instruct", "7B", "multilingual"],
        "likes": 9500,
        "size_gb": 15.2,
    },
    {
        "id": "microsoft/Phi-3.5-mini-instruct",
        "name": "Phi-3.5 Mini Instruct",
        "description": "Pequeño pero potente. ~7.6 GB. Ideal para CPU.",
        "tags": ["chat", "instruct", "3.8B", "microsoft"],
        "likes": 7000,
        "size_gb": 7.6,
    },
    {
        "id": "HuggingFaceH4/zephyr-7b-beta",
        "name": "Zephyr 7B Beta",
        "description": "Fine-tune de Mistral optimizado para ser útil y honesto. ~14 GB.",
        "tags": ["chat", "instruct", "7B", "zephyr"],
        "likes": 6000,
        "size_gb": 14.0,
    },
    {
        "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "name": "TinyLlama 1.1B Chat",
        "description": "Modelo ultrapequeño. ~2.2 GB. Perfecto para probar localmente en CPU.",
        "tags": ["chat", "1.1B", "tiny", "cpu-friendly"],
        "likes": 4500,
        "size_gb": 2.2,
    },
    {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "name": "Qwen 2.5 0.5B Instruct",
        "description": "Muy ligero, 0.5B parámetros. ~1 GB. Excelente para CPU.",
        "tags": ["chat", "instruct", "0.5B", "cpu-friendly"],
        "likes": 3000,
        "size_gb": 1.0,
    },
]


# ─── Helpers locales ──────────────────────────────────────────────────────────

def _model_key(model_id: str) -> str:
    return model_id.replace("/", "--")

def _model_local_dir(model_id: str) -> Path:
    return MODELS_CACHE_DIR / _model_key(model_id)

def _is_downloaded(model_id: str) -> bool:
    """Verifica que el directorio tiene TODOS los pesos completos del modelo."""
    d = _model_local_dir(model_id)
    if not d.exists():
        return False
    # Si hay archivos .lock, aún hay shards descargándose/incompletos
    if any(d.rglob("*.lock")):
        return False
    # Si hay archivos .incomplete (formato alternativo de HF hub)
    if any(d.rglob("*.incomplete")):
        return False
    # Revisar el índice: si existe safetensors index, verificar que todos los shards estén
    index_file = d / "model.safetensors.index.json"
    if index_file.exists():
        try:
            import json as _json
            idx = _json.loads(index_file.read_text())
            # Obtener todos los shards listados en el índice
            weight_map = idx.get("weight_map", {})
            required_shards = set(weight_map.values())
            for shard in required_shards:
                if not (d / shard).exists():
                    return False  # Shard faltante
            return len(required_shards) > 0
        except Exception:
            pass
    # Sin índice: verificar al menos un archivo de pesos
    weight_patterns = ["*.safetensors", "*.bin", "*.gguf", "*.pt", "*.litertlm", "*.tflite", "*.onnx"]
    for pat in weight_patterns:
        if any(d.rglob(pat)):
            return True
    return False

def _is_partial(model_id: str) -> bool:
    """Directorio existe pero descarga incompleta."""
    d = _model_local_dir(model_id)
    return d.exists() and not _is_downloaded(model_id)


def _dir_size_gb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / 1e9, 2)

def _get_loaded_model_key() -> Optional[str]:
    """Devuelve el key del modelo actualmente cargado (sólo uno a la vez)."""
    with _model_lock:
        if _loaded_models:
            return next(iter(_loaded_models))
    return None

def _short_model_alias(model_id_or_key: str) -> str:
    """Genera un alias corto consistente para IDs remotos y keys locales."""
    normalized = model_id_or_key.replace("--", "/")
    return normalized.split("/")[-1].split("-")[0].lower()

def _release_loaded_resources(data: dict) -> None:
    """Libera el modelo y limpia cachés de CPU/GPU."""
    try:
        import gc
        import torch

        if "model" in data:
            del data["model"]
        if "tokenizer" in data:
            del data["tokenizer"]
        if "processor" in data:
            del data["processor"]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass

def _system_info() -> dict:
    """Info básica de hardware."""
    info = {"cpu_count": os.cpu_count()}
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / 1e9, 1)
        info["ram_available_gb"] = round(mem.available / 1e9, 1)
    except ImportError:
        pass
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory
            info["vram_gb"] = round(vram / 1e9, 1)
    except ImportError:
        info["cuda_available"] = False
    return info

# ─── Hilo de descarga ─────────────────────────────────────────────────────────

class DownloadCancelled(Exception):
    pass

def _async_raise(tid, exctype):
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

def _download_thread(model_id: str, token: str):
    key = _model_key(model_id)
    local_dir = _model_local_dir(model_id)

    try:
        with _download_lock:
            _download_state[key] = {
                "status": "downloading",
                "progress": 0,
                "message": "Iniciando descarga...",
                "model_id": model_id,
            }

        logger.info(f"Starting/Resuming download: {model_id}")

        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            token=token or None,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
            local_dir_use_symlinks=False,
        )

        size = _dir_size_gb(local_dir)
        with _download_lock:
            _download_state[key] = {
                "status": "done",
                "progress": 100,
                "message": f"✓ Descarga completada ({size} GB)",
                "model_id": model_id,
                "size_gb": size,
            }
        logger.info(f"Download complete: {model_id} ({size} GB)")

    except DownloadCancelled:
        logger.info(f"Download cancelled by user: {model_id}")
        # Clean state naturally
    except Exception as e:
        logger.error(f"Download error [{model_id}]: {e}")
        # ⚠ NO borrar archivos parciales — huggingface_hub los reutiliza para reanudar.
        # Los archivos .incomplete de HF son el mecanismo de resume automático.
        size_saved = _dir_size_gb(local_dir) if local_dir.exists() else 0
        with _download_lock:
            _download_state[key] = {
                "status": "partial",
                "progress": 0,
                "message": f"Descarga interrumpida ({size_saved} GB guardados). Podés reintentar para continuar.",
                "model_id": model_id,
                "size_saved_gb": size_saved,
            }


# ─── Hilo de carga ────────────────────────────────────────────────────────────

def _load_model_thread(model_id: str, quantization: str, device: str):
    key = _model_key(model_id)
    local_dir = _model_local_dir(model_id)

    try:
        # ── Validar que la descarga está completa antes de intentar cargar ──
        if _is_partial(model_id):
            size_saved = _dir_size_gb(local_dir)
            with _model_lock:
                _download_state[key + "_load"] = {
                    "status": "error",
                    "progress": 0,
                    "message": f"Descarga incompleta ({size_saved} GB guardados). Reanudá la descarga primero.",
                    "model_id": model_id,
                }
            return

        with _model_lock:
            _download_state[key + "_load"] = {
                "status": "loading",
                "progress": 10,
                "message": "Importando transformers...",
                "model_id": model_id,
            }

        # Parche para bug de transformers leyendo la versión de gguf
        try:
            import gguf
            gguf.__version__ = "0.18.0"
        except ImportError:
            pass

        # HF Transformers / PyTorch
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            AutoModelForImageTextToText = None

        config = AutoConfig.from_pretrained(str(local_dir), trust_remote_code=True)
        model_meta = _extract_model_meta(model_id, config=config.to_dict())
        supports_vision = model_meta["supports_vision"]
        if supports_vision and AutoModelForImageTextToText is None:
            logger.warning(
                "Installed transformers does not provide AutoModelForImageTextToText; "
                "falling back to text-only loading for %s",
                model_id,
            )
            supports_vision = False
            model_meta["supports_vision"] = False
            model_meta["capability_label"] = "Solo texto"

        # Detectar device disponible respetando la preferencia del usuario
        use_cuda = torch.cuda.is_available() and device != "cpu"
        if use_cuda:
            target_device = torch.device("cuda")
            dtype = torch.float16
        else:
            target_device = torch.device("cpu")
            dtype = torch.float32
            try:
                import transformers.utils.import_utils as _tf_import_utils
                _tf_import_utils._bitsandbytes_available = False
            except Exception:
                pass

        # --- Hardware safety check ---
        sys_info = _system_info()
        vram_avail = sys_info.get("vram_gb", 0) if sys_info.get("cuda_available", False) else 0
        ram_avail = sys_info.get("ram_available_gb", 0)

        size_gb = _dir_size_gb(local_dir)
        if quantization == "8bit":
            required_gb = size_gb * 0.55
        elif quantization == "4bit":
            required_gb = size_gb * 0.3
        else:
            required_gb = size_gb

        required_gb += 1.0  # Base overhead para contexto y KV cache

        if use_cuda:
            if quantization == "none":
                if required_gb > ram_avail:
                    raise RuntimeError(f"RAM insuficiente: Requiere ~{required_gb:.1f} GB de RAM para preparar el modelo, disponible {ram_avail:.1f} GB.")
                if required_gb > vram_avail:
                    raise RuntimeError(f"VRAM insuficiente: Requiere ~{required_gb:.1f} GB de VRAM, disponible {vram_avail:.1f} GB. Usa cuantización 4bit u 8bit.")
            else:
                if required_gb > vram_avail:
                    raise RuntimeError(f"VRAM insuficiente para {quantization}: Requiere ~{required_gb:.1f} GB, disponible {vram_avail:.1f} GB.")
        else:
            # Si corre en CPU
            if quantization in ("4bit", "8bit"):
                 # bitsandbytes en CPU no está soportado oficialmente, pero si se intenta:
                 raise RuntimeError("La cuantización 4bit/8bit requiere una GPU.")
            if required_gb > ram_avail:
                raise RuntimeError(f"RAM insuficiente: Requiere ~{required_gb:.1f} GB, disponible {ram_avail:.1f} GB.")
        # -----------------------------

        # Detección de incompatibles (LiteRT, ONNX, etc) si solo están esos archivos
        litert_files = list(local_dir.rglob("*.litertlm"))
        safetensors = list(local_dir.rglob("*.safetensors"))
        if litert_files and not safetensors:
            raise RuntimeError("Este es un modelo LiteRT (.litertlm). La herramienta transformers predeterminada de este proyecto web no puede cargarlo, ¡está pensado solo para móviles y web! Necesitas la versión original safetensors.")

        with _model_lock:
            _download_state[key + "_load"]["message"] = "Cargando tokenizer/procesador..."
            _download_state[key + "_load"]["progress"] = 25

        # Detección de archivos GGUF
        gguf_files = list(local_dir.rglob("*.gguf"))
        gguf_kwargs = {}
        is_gguf = len(gguf_files) > 0
        if is_gguf:
            selected_gguf = gguf_files[0]
            if quantization == "4bit":
                q4_files = [f for f in gguf_files if "q4" in f.name.lower()]
                if q4_files:
                    selected_gguf = q4_files[0]
            elif quantization == "8bit":
                q8_files = [f for f in gguf_files if "q8" in f.name.lower()]
                if q8_files:
                    selected_gguf = q8_files[0]
            gguf_kwargs["gguf_file"] = selected_gguf.name
            logger.info(f"Detectado modelo GGUF: usando archivo {selected_gguf.name}")

        processor = None
        tokenizer = None
        if supports_vision and not is_gguf:
            processor = AutoProcessor.from_pretrained(
                str(local_dir),
                trust_remote_code=True,
            )
            tokenizer = getattr(processor, "tokenizer", None)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                str(local_dir),
                trust_remote_code=True,
                **gguf_kwargs
            )

        with _model_lock:
            _download_state[key + "_load"]["message"] = "Cargando modelo (puede tardar varios minutos)..."
            _download_state[key + "_load"]["progress"] = 40

        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        model_kwargs.update(gguf_kwargs)

        # Cuantización (requiere CUDA + bitsandbytes)
        # IMPORTANTE: Si es GGUF, ya viene cuantizado en su peso, NO podemos usar bitsandbytes
        if quantization in ("4bit", "8bit") and not is_gguf:
            try:
                from transformers import BitsAndBytesConfig
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=(quantization == "4bit"),
                    load_in_8bit=(quantization == "8bit"),
                    bnb_4bit_compute_dtype=torch.float16 if quantization == "4bit" else None,
                )
                model_kwargs["quantization_config"] = bnb_cfg
                model_kwargs["device_map"] = "auto"
                target_device = None  # quantization handles placement
            except ImportError:
                logger.warning("bitsandbytes no instalado, cargando sin cuantización")
                quantization = "none"

        if quantization == "none" or is_gguf:
            model_kwargs["torch_dtype"] = dtype  # float16 si CUDA, float32 si CPU

        try:
            model_cls = AutoModelForImageTextToText if supports_vision and not is_gguf else AutoModelForCausalLM
            model = model_cls.from_pretrained(str(local_dir), **model_kwargs)
        except Exception as load_err:
            logger.warning(f"First load attempt failed ({load_err}), retrying without dtype hints...")
            # Fallback: cargar sin optimizaciones (más lento pero compatible con todo)
            model_cls = AutoModelForImageTextToText if supports_vision and not is_gguf else AutoModelForCausalLM
            model = model_cls.from_pretrained(
                str(local_dir),
                trust_remote_code=True,
            )

        # Mover al device manualmente (evita el bug de accelerate con device_map="auto")
        if target_device is not None:
            logger.info(f"Moving model to {target_device}")
            try:
                model = model.to(target_device)
            except Exception as move_err:
                logger.warning(f"Could not move model to {target_device}: {move_err}. Staying on CPU.")

        model.eval()


        with _model_lock:
            _download_state[key + "_load"]["message"] = "Finalizando..."
            _download_state[key + "_load"]["progress"] = 90

        with _model_lock:
            _loaded_models[key] = {
                "tokenizer": tokenizer,
                "processor": processor,
                "model": model,
                "model_id": model_id,
                "quantization": quantization,
                "supports_vision": supports_vision,
                "model_type": model_meta["model_type"],
            }
            _download_state[key + "_load"] = {
                "status": "loaded",
                "progress": 100,
                "message": "✓ Modelo cargado y listo" + (" (vision + texto)" if supports_vision else ""),
                "model_id": model_id,
            }
        logger.info(f"Model loaded: {model_id}")

    except Exception as e:
        logger.error(f"Load error [{model_id}]: {e}")
        with _model_lock:
            _download_state[key + "_load"] = {
                "status": "error",
                "progress": 0,
                "message": str(e),
                "model_id": model_id,
            }


# ──────────────────────────────────────────────────────────────────────────────
# ENDPOINTS – Modelos locales
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return RedirectResponse(url="/app", status_code=307)


@app.get("/health")
async def health():
    return {"status": "ok", "version": APP_VERSION, "mode": LLMFRONT_MODE, "local_support": LOCAL_MODE}


@app.get("/config")
async def get_config():
    """Configuración activa del servidor (modo, features habilitadas)."""
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
    }


@app.get("/system/info")
async def system_info():
    return _system_info()


@app.get("/models/featured")
async def get_featured_models():
    """Modelos curados con info de tamaño y estado local."""
    models = []
    for m in FEATURED_MODELS:
        entry = dict(m)
        entry.update(_extract_model_meta(
            m["id"],
            tags=m.get("tags", []),
            description=m.get("description", ""),
        ))
        entry["is_downloaded"] = _is_downloaded(m["id"])
        entry["is_partial"] = _is_partial(m["id"])
        key = _model_key(m["id"])
        with _model_lock:
            entry["is_loaded"] = key in _loaded_models
        if entry["is_downloaded"]:
            entry["local_size_gb"] = _dir_size_gb(_model_local_dir(m["id"]))
        models.append(entry)
    return {"models": models}


@app.get("/models/trending")
async def get_trending_models():
    """Fetch top trending models from Hugging Face Hub."""
    try:
        from huggingface_hub import list_models
        # Fetch top 10 trending text-generation models
        hf_models = list_models(sort="trending", limit=10, filter="text-generation", cardData=True)
        
        models = []
        for m in hf_models:
            model_id = m.modelId
            entry = {
                "id": model_id,
                "name": model_id.split('/')[-1].replace('-', ' ').title(),
                "description": f"Modelo tendencia en Hugging Face. Creado por {model_id.split('/')[0]}.",
                "tags": getattr(m, 'tags', [])[:4],
                "likes": getattr(m, 'likes', 0),
                "is_downloaded": _is_downloaded(model_id),
                "is_partial": _is_partial(model_id),
                "is_trending": True
            }
            # Try to guess size or get from siblings
            entry["size_gb"] = -1 # Default unknown
            models.append(entry)
            
        return {"models": models}
    except Exception as e:
        logger.error(f"Error fetching trending models: {e}")
        return {"models": [], "error": str(e)}


def _guess_size_b(name: str, tags: list) -> float:
    for tag in tags:
        m = re.search(r'(?i)(\d+(?:\.\d+)?)b', tag)
        if m:
            return float(m.group(1))
    m = re.search(r'(?i)[_-](\d+(?:\.\d+)?)b[_-]', name)
    if m:
        return float(m.group(1))
    return -1


def _extract_model_meta(model_id: str, tags: Optional[list] = None, pipeline_tag: str = "", description: str = "", config: Optional[dict] = None) -> dict:
    tags = tags or []
    config = config or {}
    haystack = " ".join([
        model_id or "",
        pipeline_tag or "",
        description or "",
        config.get("model_type", "") or "",
        " ".join(config.get("architectures", []) or []),
        " ".join(tags),
    ]).lower()
    vision_keywords = (
        "llava", "vision", "vlm", "image-text-to-text", "image text to text",
        "idefics", "paligemma", "qwen2-vl", "qwen-vl", "smolvlm",
        "cambrian", "bunny", "internvl", "minicpm-v", "glm-4v"
    )
    supports_vision = any(keyword in haystack for keyword in vision_keywords)
    return {
        "supports_vision": supports_vision,
        "supports_local_chat": True,
        "capability_label": "Vision + Texto" if supports_vision else "Solo texto",
        "model_type": config.get("model_type", ""),
        "architectures": config.get("architectures", []) or [],
        "pipeline_tag": pipeline_tag or "",
    }


def _read_local_config(local_dir: Path) -> dict:
    config_file = local_dir / "config.json"
    if not config_file.exists():
        return {}
    try:
        return json.loads(config_file.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _matches_query(query: str, *values: str) -> bool:
    if not query.strip():
        return True
    haystack = " ".join(v for v in values if v).lower()
    terms = [t for t in re.split(r"\s+", query.lower().strip()) if t]
    return all(term in haystack for term in terms)

def _passes_size_filter(model_id: str, tags: list, size_filter: str) -> bool:
    if size_filter == "any":
        return True
    size_b = _guess_size_b(model_id, tags)
    if size_b <= 0:
        return True
    if size_filter == "small":
        return size_b <= 3.5
    if size_filter == "medium":
        return 3.5 < size_b < 9.5
    if size_filter == "large":
        return size_b >= 9.5
    return True

def _provider_search_result(
    model_id: str,
    name: str,
    description: str = "",
    tags: Optional[list] = None,
    likes: int = 0,
    downloads: int = 0,
    pipeline_tag: str = "",
    extra_meta: Optional[dict] = None,
) -> dict:
    tags = tags or []
    extra_meta = extra_meta or {}
    meta = _extract_model_meta(
        model_id,
        tags=tags,
        pipeline_tag=pipeline_tag,
        description=description,
        config=extra_meta,
    )
    return {
        "id": model_id,
        "name": name or model_id,
        "description": description,
        "tags": tags[:6],
        "likes": likes,
        "downloads": downloads,
        "pipeline_tag": pipeline_tag,
        "is_downloaded": _is_downloaded(model_id),
        "is_partial": _is_partial(model_id),
        **meta,
    }

def _search_openrouter_models(query: str, size_filter: str, limit: int) -> tuple[list, str]:
    resp = requests.get("https://openrouter.ai/api/v1/models", timeout=20)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    models = []
    for item in data:
        model_id = item.get("id", "")
        name = item.get("name") or item.get("canonical_slug") or model_id
        description = item.get("description", "") or ""
        architecture = item.get("architecture", {}) or {}
        modality = architecture.get("modality", "") or ""
        input_modalities = architecture.get("input_modalities", []) or []
        tags = [t for t in [modality, *input_modalities] if t]
        if not _matches_query(query, model_id, name, description, " ".join(tags)):
            continue
        if not _passes_size_filter(model_id, tags, size_filter):
            continue
        models.append(_provider_search_result(
            model_id=model_id,
            name=name,
            description=description,
            tags=tags,
            pipeline_tag="conversational",
        ))
        if len(models) >= limit:
            break
    return models, "openrouter"

def _search_together_models(query: str, size_filter: str, limit: int, api_key: str) -> tuple[list, str]:
    if not api_key:
        raise HTTPException(status_code=400, detail="Together AI requiere API key para listar modelos.")
    resp = requests.get(
        "https://api.together.xyz/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    models = []
    for item in data:
        model_id = item.get("id", "")
        name = item.get("display_name") or model_id
        description = f"Tipo: {item.get('type', 'unknown')}"
        tags = [item.get("type", "")]
        if item.get("organization"):
            tags.append(item["organization"])
        if item.get("context_length"):
            tags.append(f"{item['context_length']}ctx")
        if not _matches_query(query, model_id, name, description, " ".join(tags)):
            continue
        if not _passes_size_filter(model_id, tags, size_filter):
            continue
        models.append(_provider_search_result(
            model_id=model_id,
            name=name,
            description=description,
            tags=tags,
            pipeline_tag="conversational" if item.get("type") == "chat" else "text-generation",
        ))
        if len(models) >= limit:
            break

    # Probe exact query if it looks like a full model ID
    if query and "/" in query and not any(m["id"] == query for m in models):
        try:
            probe = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": query, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1},
                timeout=10,
            )
            if probe.ok:
                models.insert(0, _provider_search_result(
                    model_id=query,
                    name=query.split("/")[-1],
                    description="Custom Model (Probed via direct API call)",
                    tags=["custom", "together"],
                    pipeline_tag="conversational",
                ))
        except Exception:
            pass

    return models, "together"

def _search_fireworks_models(query: str, size_filter: str, limit: int, api_key: str) -> tuple[list, str]:
    if not api_key:
        raise HTTPException(status_code=400, detail="Fireworks AI requiere API key para listar modelos.")
    resp = requests.get(
        "https://api.fireworks.ai/v1/accounts/fireworks/models",
        headers={"Authorization": f"Bearer {api_key}"},
        params={"filter": "supports_serverless=true", "pageSize": min(limit * 3, 200)},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json().get("models", [])
    models = []

    def _is_invocable_fireworks_model(model_id: str) -> bool:
        try:
            probe = requests.post(
                "https://api.fireworks.ai/inference/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "stream": False,
                },
                timeout=12,
            )
            return probe.ok
        except Exception:
            return False

    for item in data:
        status = item.get("status", {}) or {}
        if status.get("code") and status.get("code") != "OK":
            continue
        if item.get("supportsServerless") is False:
            continue
        model_id = item.get("name", "")
        name = item.get("displayName") or model_id
        description = item.get("description", "") or ""
        base = item.get("baseModelDetails", {}) or {}
        tags = [t for t in [base.get("modelType", ""), base.get("parameterCount", "")] if t]
        model_type = (base.get("modelType", "") or "").lower()
        if any(term in model_type for term in ["embed", "rerank", "audio", "vision"]):
            continue
        if item.get("huggingFaceUrl"):
            tags.append("huggingface")
        if not _matches_query(query, model_id, name, description, " ".join(tags)):
            continue
        if not _passes_size_filter(model_id, tags, size_filter):
            continue
        if not _is_invocable_fireworks_model(model_id):
            continue
        models.append(_provider_search_result(
            model_id=model_id,
            name=name,
            description=description,
            tags=tags,
            pipeline_tag="conversational",
            extra_meta={"model_type": base.get("modelType", "")},
        ))
        if len(models) >= limit:
            break
            
    # Probe exact query if it looks like a full model ID
    if query and "/" in query and not any(m["id"] == query for m in models):
        if _is_invocable_fireworks_model(query):
            models.insert(0, _provider_search_result(
                model_id=query,
                name=query.split("/")[-1],
                description="Custom Model (Probed via direct API call)",
                tags=["custom", "fireworks"],
                pipeline_tag="conversational",
            ))

    return models, "fireworks"

def _search_baseten_models(query: str, size_filter: str, limit: int, api_key: str) -> tuple[list, str]:
    if not api_key:
        raise HTTPException(status_code=400, detail="Baseten requiere API key para listar modelos.")
    resp = requests.get(
        "https://inference.baseten.co/v1/models",
        headers={"Authorization": f"Api-Key {api_key}"},
        timeout=20,
    )
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data") or payload.get("models") or []
    models = []
    for item in data:
        model_id = item.get("id", "")
        name = item.get("name") or model_id
        description = item.get("description", "") or ""
        tags = []
        architecture = item.get("architecture", {}) or {}
        if architecture.get("modality"):
            tags.append(architecture["modality"])
        if item.get("top_provider"):
            tags.append(item["top_provider"].get("name", ""))
        if item.get("context_length"):
            tags.append(f"{item['context_length']}ctx")
        if not _matches_query(query, model_id, name, description, " ".join(tags)):
            continue
        if not _passes_size_filter(model_id, tags, size_filter):
            continue
        models.append(_provider_search_result(
            model_id=model_id,
            name=name,
            description=description,
            tags=tags,
            pipeline_tag="conversational",
        ))
        if len(models) >= limit:
            break

    # Probe exact query if it looks like a full model ID
    if query and not any(m["id"] == query for m in models):
        try:
            probe = requests.post(
                "https://inference.baseten.co/v1/chat/completions",
                headers={"Authorization": f"Api-Key {api_key}", "Content-Type": "application/json"},
                json={"model": query, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1},
                timeout=10,
            )
            if probe.ok:
                models.insert(0, _provider_search_result(
                    model_id=query,
                    name=query,
                    description="Custom Model (Probed via direct API call)",
                    tags=["custom", "baseten"],
                    pipeline_tag="conversational",
                ))
        except Exception:
            pass

    return models, "baseten"

def _check_hf_api_availability(model_id: str, token: str) -> dict:
    if not token:
        return {
            "status": "unknown",
            "label": "Sin verificar",
            "mode": None,
            "detail": "Se necesita un HF Token para comprobar compatibilidad API.",
        }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    def _classify_probe_response(resp) -> tuple[str, str]:
        text = (resp.text or "")[:280]
        lowered = text.lower()
        if resp.ok:
            return "available", text
        if any(term in lowered for term in [
            "not a chat model",
            "model_not_supported",
            "not supported",
            "unsupported",
            "text-generation is not supported",
        ]):
            return "unsupported", text
        if resp.status_code in (401, 403):
            return "auth", text
        if resp.status_code in (408, 409, 424, 425, 429, 500, 502, 503, 504):
            return "transient", text
        return "unknown", text

    chat_url = f"https://api-inference.huggingface.co/models/{model_id}/v1/chat/completions"
    try:
        resp = requests.post(
            chat_url,
            headers=headers,
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Hola"}],
                "max_tokens": 1,
                "stream": False,
                "temperature": 0.1,
            },
            timeout=20,
        )
        chat_status, chat_error = _classify_probe_response(resp)
        if chat_status == "available":
            return {
                "status": "available",
                "label": "Disponible",
                "mode": "chat",
                "detail": "HF responde por chat completions.",
            }
    except Exception as e:
        chat_status = "unknown"
        chat_error = str(e)

    text_url = f"https://api-inference.huggingface.co/models/{model_id}"
    try:
        resp = requests.post(
            text_url,
            headers=headers,
            json={
                "inputs": "Hola",
                "parameters": {
                    "max_new_tokens": 1,
                    "return_full_text": False,
                    "temperature": 0.1,
                },
                "options": {
                    "wait_for_model": False,
                    "use_cache": False,
                },
            },
            timeout=20,
        )
        text_status, text_error = _classify_probe_response(resp)
        if text_status == "available":
            return {
                "status": "available",
                "label": "Disponible",
                "mode": "text-generation",
                "detail": "HF responde por text-generation.",
            }
    except Exception as e:
        text_status = "unknown"
        text_error = str(e)

    if chat_status == "auth" or text_status == "auth":
        return {
            "status": "unknown",
            "label": "Sin verificar",
            "mode": None,
            "detail": "Token inválido o sin permisos para comprobar compatibilidad API.",
        }

    if chat_status == "unsupported" and text_status == "unsupported":
        return {
            "status": "unavailable",
            "label": "No disponible",
            "mode": None,
            "detail": text_error or chat_error or "HF no expone este modelo en la Inference API actual.",
        }

    if chat_status == "unsupported":
        return {
            "status": "available",
            "label": "Disponible",
            "mode": "text-generation",
            "detail": "No responde por chat, pero podria funcionar por text-generation o por fallback.",
        }

    if text_status == "unsupported":
        return {
            "status": "available",
            "label": "Disponible",
            "mode": "chat",
            "detail": "No responde por text-generation, pero podria funcionar por chat completions.",
        }

    return {
        "status": "unknown",
        "label": "Sin verificar",
        "mode": None,
        "detail": text_error or chat_error or "No se pudo confirmar la compatibilidad API en este momento.",
    }

def _check_openai_provider_model_availability(provider: str, model_id: str, api_key: str) -> dict:
    base_urls = {
        "groq": "https://api.groq.com/openai/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "together": "https://api.together.xyz/v1",
        "deepinfra": "https://api.deepinfra.com/v1/openai",
        "fireworks": "https://api.fireworks.ai/inference/v1",
        "baseten": "https://inference.baseten.co/v1",
    }
    base_url = base_urls.get(provider)
    if not api_key or not base_url:
        return {
            "status": "unknown",
            "label": "Sin verificar",
            "mode": None,
            "detail": f"No hay suficiente informacion para verificar {provider.upper()}.",
        }

    headers = {
        "Authorization": f"Bearer {api_key}" if provider != "baseten" else f"Api-Key {api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 1,
                "temperature": 0.0,
                "stream": False,
            },
            timeout=15,
        )
        msg = (resp.text or "")[:280]
        lowered = msg.lower()
        if resp.ok:
            return {
                "status": "available",
                "label": "Disponible",
                "mode": "chat",
                "detail": f"{provider.upper()} responde por chat completions.",
            }
        if "model_not_found" in lowered or "does not exist" in lowered or "not found" in lowered:
            return {
                "status": "unavailable",
                "label": "No disponible",
                "mode": None,
                "detail": f"{provider.upper()} no reconoce este model id para chat.",
            }
        if "embedding" in lowered or "not a chat model" in lowered or "not supported" in lowered:
            return {
                "status": "unavailable",
                "label": "No disponible",
                "mode": None,
                "detail": f"{provider.upper()} indica que el modelo no es apto para chat.",
            }
        return {
            "status": "unknown",
            "label": "Sin verificar",
            "mode": None,
            "detail": msg or f"No se pudo confirmar compatibilidad de {provider.upper()} en este momento.",
        }
    except Exception as e:
        return {
            "status": "unknown",
            "label": "Sin verificar",
            "mode": None,
            "detail": str(e),
        }

@app.post("/models/search")
async def search_models(req: ModelSearchRequest):
    token = req.hf_token or HF_TOKEN
    search_query = req.query
    
    # 🧠 Búsqueda Semántica Asistida por IA (Groq)
    if req.use_ai_search and not GROQ_AI_SEARCH_ENABLED:
        logger.info("Búsqueda IA solicitada pero GROQ_AI_SEARCH_ENABLED=false, ignorando.")
        req = req.model_copy(update={"use_ai_search": False})
    if req.use_ai_search and GROQ_API_KEY and search_query:
        try:
            groq_prompt = (
                f"El usuario quiere buscar un modelo de inteligencia artificial open-source "
                f"con esta intención: '{search_query}'. "
                f"Dime 3 palabras clave perfectas para buscar en HuggingFace o 2 IDs completos de los mejores modelos exactos actuales. "
                f"REGLA OBLIGATORIA: Responde SOLO con una cadena de texto separada por espacios para rellenar la barra de búsqueda. NADA MÁS."
            )
            groq_res = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": groq_prompt}],
                    "temperature": 0.2,
                    "max_tokens": 50
                },
                timeout=5
            )
            groq_res.raise_for_status()
            ai_reply = groq_res.json()["choices"][0]["message"]["content"].strip()
            # Quitamos comillas si la IA listó algo
            ai_reply = ai_reply.replace('"', '').replace("'", "")
            search_query = f"{ai_reply}"
            logger.info(f"AI Search Query transformed: '{req.query}' -> '{search_query}'")
        except Exception as e:
            logger.warning(f"Groq AI Search failed, falling back to standard: {e}")

    provider = (req.provider or "hf").lower()
    provider_api_keys = {
        "groq": GROQ_API_KEY,
        "openrouter": os.getenv("OPENROUTER_API_KEY", ""),
        "together": TOGETHER_API_KEY,
        "deepinfra": DEEPINFRA_API_KEY,
        "fireworks": FIREWORKS_API_KEY,
        "baseten": BASETEN_API_KEY,
    }
    effective_api_key = req.api_key or provider_api_keys.get(provider, "")

    try:
        if provider == "openrouter":
            models, source = _search_openrouter_models(search_query, req.size_filter, req.limit)
            return {"models": models, "total": len(models), "source": source}
        if provider == "together":
            models, source = _search_together_models(search_query, req.size_filter, req.limit, effective_api_key)
            return {"models": models, "total": len(models), "source": source}
        if provider == "fireworks":
            models, source = _search_fireworks_models(search_query, req.size_filter, req.limit, effective_api_key)
            return {"models": models, "total": len(models), "source": source}
        if provider == "baseten":
            models, source = _search_baseten_models(search_query, req.size_filter, req.limit, effective_api_key)
            return {"models": models, "total": len(models), "source": source}

        source = "huggingface"
        if provider == "deepinfra":
            source = "huggingface-fallback-for-deepinfra"

        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        fetch_limit = req.limit * 4 if req.size_filter != "any" else req.limit
        params = {"search": search_query, "filter": req.task, "sort": "likes", "direction": -1, "limit": fetch_limit, "full": False}
        resp = requests.get("https://huggingface.co/api/models", params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models = []
        for m in data:
            mid = m.get("modelId", m.get("id", ""))
            tags = m.get("tags", [])
            description = (m.get("cardData", {}) or {}).get("description", "") or ""
            meta = _extract_model_meta(
                mid,
                tags=tags,
                pipeline_tag=m.get("pipeline_tag", ""),
                description=description,
            )
            
            if not _passes_size_filter(mid, tags, req.size_filter):
                continue

            models.append({
                "id": mid,
                "name": mid,
                "description": description,
                "tags": tags[:5],
                "likes": m.get("likes", 0),
                "downloads": m.get("downloads", 0),
                "pipeline_tag": m.get("pipeline_tag", ""),
                "is_downloaded": _is_downloaded(mid),
                "is_partial": _is_partial(mid),
                **meta,
            })
            if len(models) >= req.limit:
                break
        return {"models": models, "total": len(models), "source": source}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/size/{model_id:path}")
async def get_model_size(model_id: str):
    try:
        resp = requests.get(f"https://huggingface.co/api/models/{model_id}", headers={"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            st = data.get("safetensors", {})
            total_bytes = st.get("total")
            if total_bytes:
                return {"size_gb": round(total_bytes / 1e9, 2)}
        return {"size_gb": None}
    except Exception:
        return {"size_gb": None}

@app.post("/models/api-check")
async def check_model_api(req: ModelApiCheckRequest):
    provider = (req.provider or "hf").lower()
    if provider == "hf":
        token = req.hf_token or HF_TOKEN
        return _check_hf_api_availability(req.model_id, token)

    env_api_keys = {
        "groq": GROQ_API_KEY,
        "openrouter": os.getenv("OPENROUTER_API_KEY", ""),
        "together": TOGETHER_API_KEY,
        "deepinfra": DEEPINFRA_API_KEY,
        "fireworks": FIREWORKS_API_KEY,
        "baseten": BASETEN_API_KEY,
    }
    effective_api_key = req.api_key or env_api_keys.get(provider, "")
    return _check_openai_provider_model_availability(provider, req.model_id, effective_api_key)



if LOCAL_MODE:
    @app.get("/models/local")
    async def list_local_models():
        """Lista todos los modelos descargados localmente."""
        local = []
        for d in MODELS_CACHE_DIR.iterdir():
            if d.is_dir():
                model_id = d.name.replace("--", "/", 1)
                key = d.name
                config = _read_local_config(d)
                meta = _extract_model_meta(model_id, config=config)
                with _model_lock:
                    is_loaded = key in _loaded_models
                size = _dir_size_gb(d)
                is_complete = _is_downloaded(model_id)
                is_loading_now = False
                with _download_lock:
                    dl_state = _download_state.get(key, {})
                    is_loading_now = dl_state.get("status") == "downloading"
                local.append({
                    "id": model_id,
                    "name": model_id.split("/")[-1],
                    "local_dir": str(d),
                    "size_gb": size,
                    "is_loaded": is_loaded,
                    "is_complete": is_complete,
                    "is_partial": not is_complete and any(d.iterdir()),
                    "is_downloading": is_loading_now,
                    **meta,
                })
        return {"models": local}


if LOCAL_MODE:
    @app.post("/models/download")
    async def start_download(req: DownloadRequest):
        """Inicia (o reanuda) la descarga de un modelo en background."""
        key = _model_key(req.model_id)
        with _download_lock:
            existing = _download_state.get(key, {})
            if existing.get("status") == "downloading":
                return {"status": "already_downloading", "message": "Ya se está descargando este modelo"}

        is_partial = _is_partial(req.model_id)
        size_saved = _dir_size_gb(_model_local_dir(req.model_id)) if is_partial else 0
        with _download_lock:
            _download_state[key] = {
                "status": "downloading",
                "progress": 0,
                "message": (
                    f"Reanudando descarga ({size_saved} GB ya guardados)..."
                    if is_partial else
                    "Iniciando descarga..."
                ),
                "model_id": req.model_id,
                "thread_id": None,
            }
            if is_partial:
                _download_state[key]["size_saved_gb"] = size_saved
                logger.info(f"Resuming partial download: {req.model_id} ({size_saved} GB saved)")

        token = req.hf_token or HF_TOKEN
        
        def run_thread():
            with _download_lock:
                _download_state[key]["thread_id"] = threading.get_ident()
            _download_thread(req.model_id, token)

        t = Thread(target=run_thread, daemon=True)
        t.start()
        return {"status": "started" if not is_partial else "resuming", "model_id": req.model_id}

if LOCAL_MODE:
    @app.post("/models/download/cancel")
    async def cancel_download(model_id: str):
        key = _model_key(model_id)
        with _download_lock:
            state = _download_state.get(key)
            if state and state.get("status") == "downloading" and "thread_id" in state:
                tid = state["thread_id"]
                try:
                    _async_raise(tid, DownloadCancelled)
                    state["status"] = "cancelled"
                    state["message"] = "Descarga cancelada"
                    return {"status": "cancelled"}
                except Exception as e:
                    return {"status": "error", "message": str(e)}
        return {"status": "not_downloading"}



if LOCAL_MODE:
  @app.get("/models/download/{model_id:path}/progress")
  async def download_progress_sse(model_id: str):
    """SSE stream del progreso de descarga y carga."""
    async def generate():
        key = _model_key(model_id)
        load_key = key + "_load"
        prev = None
        timeout = 0
        while timeout < 3600:  # max 1 hora
            await asyncio.sleep(1.5)
            with _download_lock:
                dl = dict(_download_state.get(key, {}))
            with _model_lock:
                ld = dict(_download_state.get(load_key, {}))

            # Estimar progreso de descarga por tamaño del directorio
            local_dir = _model_local_dir(model_id)
            if local_dir.exists() and dl.get("status") == "downloading":
                size_now = _dir_size_gb(local_dir)
                dl["size_downloaded_gb"] = size_now

            combined = {"download": dl, "load": ld}
            if combined != prev:
                yield f"data: {json.dumps(combined)}\n\n"
                prev = combined

            status = dl.get("status")
            load_status = ld.get("status")
            if status in ("done", "error", "cancelled") and load_status in ("loaded", "error", ""):
                if load_status not in ("loading",):
                    break
            timeout += 1.5

        yield f"data: {json.dumps({'_end': True})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


if LOCAL_MODE:
    @app.get("/models/download/{model_id:path}/status")
    async def download_status(model_id: str):
        """Estado puntual de descarga/carga de un modelo."""
        key = _model_key(model_id)
        with _download_lock:
            dl = dict(_download_state.get(key, {"status": "idle"}))
        with _model_lock:
            ld = dict(_download_state.get(key + "_load", {}))
            is_loaded = key in _loaded_models

        local_dir = _model_local_dir(model_id)
        if local_dir.exists() and dl.get("status") == "downloading":
            dl["size_downloaded_gb"] = _dir_size_gb(local_dir)

        return {
            "download": dl,
            "load": ld,
            "is_downloaded": _is_downloaded(model_id),
            "is_loaded": is_loaded,
        }


if LOCAL_MODE:
    @app.post("/models/load")
    async def load_model(req: LoadModelRequest):
        """Carga un modelo local en memoria para inferencia."""
        if not _is_downloaded(req.model_id):
            raise HTTPException(status_code=404, detail="El modelo no está descargado")

        key = _model_key(req.model_id)

        with _model_lock:
            if key in _loaded_models:
                return {"status": "already_loaded", "message": "Modelo ya cargado"}
            if _loaded_models:
                old_key = next(iter(_loaded_models))
                old_data = _loaded_models.pop(old_key)
                _release_loaded_resources(old_data)
                logger.info(f"Unloaded previous model: {old_key}")

        t = Thread(target=_load_model_thread, args=(req.model_id, req.quantization, req.device), daemon=True)
        t.start()
        return {"status": "loading", "model_id": req.model_id}


if LOCAL_MODE:
    @app.post("/models/unload")
    async def unload_model(model_id: str):
        """Descarga el modelo de memoria RAM/VRAM."""
        key = _model_key(model_id)
        with _model_lock:
            if key not in _loaded_models:
                return {"status": "not_loaded"}
            data = _loaded_models.pop(key)
        _release_loaded_resources(data)
        return {"status": "unloaded", "model_id": model_id}


if LOCAL_MODE:
    @app.delete("/models/local/{model_id:path}")
    async def delete_local_model(model_id: str):
        """Elimina un modelo del disco."""
        key = _model_key(model_id)
        with _model_lock:
            if key in _loaded_models:
                raise HTTPException(400, "Desactivá el modelo antes de eliminarlo")
        local_dir = _model_local_dir(model_id)
        if not local_dir.exists():
            raise HTTPException(404, "Modelo no encontrado")
        shutil.rmtree(local_dir, ignore_errors=True)
        with _download_lock:
            _download_state.pop(key, None)
        return {"status": "deleted", "model_id": model_id}


# ──────────────────────────────────────────────────────────────────────────────
# CHAT – Inferencia API (cloud)
# ──────────────────────────────────────────────────────────────────────────────

async def stream_hf_api(req: ChatRequest) -> AsyncGenerator[str, None]:
    token = req.hf_token or HF_TOKEN
    if not token:
        yield f"data: {json.dumps({'error': 'Se necesita un HF Token para usar la Inference API.'})}\n\n"
        return

    client = InferenceClient(token=token)
    messages = []
    if req.system_prompt:
        messages.append({"role": "system", "content": req.system_prompt})
    
    # Construir historial multimodal si hay imágenes
    user_msgs_count = sum(1 for m in req.messages if m.role == "user")
    current_user_msg_idx = 0
    
    for msg in req.messages:
        if msg.role == "user":
            current_user_msg_idx += 1
            if current_user_msg_idx == user_msgs_count and req.images:
                content = []
                for img in req.images:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{img.mime_type};base64,{img.base64}"}
                    })
                content.append({"type": "text", "text": msg.content or "Describí las imágenes."})
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": msg.role, "content": msg.content})
        else:
            messages.append({"role": msg.role, "content": msg.content})

    def _plain_text_prompt() -> str:
        parts = []
        if req.system_prompt:
            parts.append(f"[SYSTEM]\n{req.system_prompt}")
        for msg in req.messages:
            if msg.role == "user":
                parts.append(f"[USER]\n{msg.content}")
            elif msg.role == "assistant":
                parts.append(f"[ASSISTANT]\n{msg.content}")
            else:
                parts.append(f"[{msg.role.upper()}]\n{msg.content}")
        parts.append("[ASSISTANT]\n")
        return "\n\n".join(parts)

    def _extract_hf_chunk_text(data: dict) -> str:
        choices = data.get("choices") or []
        if not choices:
            return ""
        choice = choices[0] or {}
        delta = choice.get("delta") or {}
        message = choice.get("message") or {}
        return (
            delta.get("content")
            or choice.get("text")
            or message.get("content")
            or ""
        )

    try:
        prompt_tokens = len(str(messages)) // 4
        completion_tokens = 0
        stream = client.chat_completion(
            model=req.model,
            messages=messages,
            max_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                completion_tokens += 1
                yield f"data: {json.dumps({'token': chunk.choices[0].delta.content})}\n\n"
            await asyncio.sleep(0)
        
        logger.info(f"API Stream done. Prompt: {prompt_tokens}, Comp: {completion_tokens}")
        yield f"data: {json.dumps({'done': True, 'tokens': {'prompt': prompt_tokens, 'completion': completion_tokens}})}\n\n"
    except Exception as e:
        logger.error(f"Error in stream_hf_api: {e}")
        msg = str(e)

        if "not a chat model" in msg.lower() or "model_not_supported" in msg.lower():
            try:
                prompt_tokens = len(str(messages)) // 4
                completion_tokens = 0
                hf_resp = requests.post(
                    f"https://api-inference.huggingface.co/models/{req.model}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": req.model,
                        "messages": messages,
                        "max_tokens": req.max_new_tokens,
                        "temperature": req.temperature,
                        "top_p": req.top_p,
                        "stream": True,
                    },
                    stream=True,
                    timeout=60,
                )
                hf_resp.raise_for_status()

                for raw_line in hf_resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    data = json.loads(payload)
                    token_text = _extract_hf_chunk_text(data)
                    if token_text:
                        completion_tokens += 1
                        yield f"data: {json.dumps({'token': token_text})}\n\n"
                    await asyncio.sleep(0)

                logger.info(
                    f"HF raw /v1/chat/completions fallback done. Prompt: {prompt_tokens}, Comp: {completion_tokens}"
                )
                yield f"data: {json.dumps({'done': True, 'tokens': {'prompt': prompt_tokens, 'completion': completion_tokens}})}\n\n"
                return
            except Exception as raw_chat_err:
                logger.error(f"Fallback to raw HF chat endpoint failed: {raw_chat_err}")
                msg = str(raw_chat_err)

        if ("not a chat model" in msg.lower() or "model_not_supported" in msg.lower()) and not req.images:
            try:
                prompt = _plain_text_prompt()
                prompt_tokens = len(prompt) // 4
                completion_tokens = 0
                stream = client.text_generation(
                    prompt,
                    model=req.model,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    stream=True,
                )
                for chunk in stream:
                    token_text = chunk if isinstance(chunk, str) else getattr(chunk, "token", None)
                    if hasattr(token_text, "text"):
                        token_text = token_text.text
                    if token_text:
                        completion_tokens += 1
                        yield f"data: {json.dumps({'token': token_text})}\n\n"
                    await asyncio.sleep(0)

                logger.info(f"HF text_generation fallback done. Prompt: {prompt_tokens}, Comp: {completion_tokens}")
                yield f"data: {json.dumps({'done': True, 'tokens': {'prompt': prompt_tokens, 'completion': completion_tokens}})}\n\n"
                return
            except Exception as gen_err:
                logger.error(f"Fallback to text_generation failed: {gen_err}")
                msg = str(gen_err)
        
        # Fallback a generación de imágenes si la API no soporta "conversational"
        if "conversational" in msg.lower() or "text-generation" in msg.lower() or "task" in msg.lower() or "text2text-generation" in msg.lower():
            try:
                prompt = ""
                for m in reversed(messages):
                    if m["role"] == "user":
                        if isinstance(m["content"], list):
                            for p in m["content"]:
                                if isinstance(p, dict) and p.get("type") == "text":
                                    prompt = p.get("text", "")
                                    break
                        else:
                            prompt = str(m["content"])
                        break
                if not prompt: prompt = "A random image"
                
                logger.info(f"Intentando fallback a text_to_image con prompt: {prompt}")
                img = client.text_to_image(prompt, model=req.model)
                import base64
                from io import BytesIO
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                md_image = f"![Generada por {req.model}](data:image/jpeg;base64,{img_str})\n"
                yield f"data: {json.dumps({'token': md_image})}\n\n"
                yield f"data: {json.dumps({'done': True, 'tokens': {'prompt': 0, 'completion': 0}})}\n\n"
                return
            except Exception as img_err:
                logger.error(f"Fallback to image failed: {img_err}")
                pass # Continuar con los errores normales
                
        if "403" in msg or "401" in msg:
            msg = "Token inválido o sin acceso al modelo."
        elif "404" in msg:
            msg = "Modelo no disponible en Inference API. Intentá descargarlo localmente."
        elif "503" in msg or "loading" in msg.lower():
            msg = "El modelo está cargando en HF. Intentá en unos segundos."
        elif "not a chat model" in msg.lower() or "model_not_supported" in msg.lower():
            msg = (
                "Hugging Face no expone este modelo como chat en Inference API. "
                "Probá usarlo en modo local o elegí otro proveedor/modelo."
            )
        elif "text-generation" in msg.lower() and "not supported" in msg.lower():
            msg = (
                "Hugging Face no expone este modelo ni como chat ni como text-generation "
                "en Inference API. Probá usarlo en modo local."
            )
        elif "not supported" in msg.lower():
            msg = f"Este modelo no está soportado por la Inference API en este modo. Detalle: {msg}"
        yield f"data: {json.dumps({'error': msg})}\n\n"

async def stream_openai_compatible_api(req: ChatRequest) -> AsyncGenerator[str, None]:
    env_api_keys = {
        "groq": GROQ_API_KEY,
        "openrouter": os.getenv("OPENROUTER_API_KEY", ""),
        "together": TOGETHER_API_KEY,
        "deepinfra": DEEPINFRA_API_KEY,
        "fireworks": FIREWORKS_API_KEY,
        "baseten": BASETEN_API_KEY,
    }
    effective_api_key = req.api_key or env_api_keys.get(req.provider, "")

    if not effective_api_key:
        yield f"data: {json.dumps({'error': f'Se necesita una API Key para usar {req.provider.upper()}.'})}\n\n"
        return
        
    base_urls = {
        "groq": "https://api.groq.com/openai/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "together": "https://api.together.xyz/v1",
        "deepinfra": "https://api.deepinfra.com/v1/openai",
        "fireworks": "https://api.fireworks.ai/inference/v1",
        "baseten": "https://inference.baseten.co/v1",
    }
    base_url = base_urls.get(req.provider)
    if not base_url:
        yield f"data: {json.dumps({'error': 'Proveedor no válido.'})}\n\n"
        return

    client = AsyncOpenAI(api_key=effective_api_key, base_url=base_url)
    messages = []
    if req.system_prompt:
        messages.append({"role": "system", "content": req.system_prompt})
    
    # Procesar mensajes con soporte básico para OpenRouter vision
    for msg in req.messages:
        if msg.role == "user" and msg == req.messages[-1] and req.images:
            content = [{"type": "text", "text": msg.content or "Describe las imágenes"}]
            for img in req.images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{img.mime_type};base64,{img.base64}"}
                })
            messages.append({"role": msg.role, "content": content})
        else:
            messages.append({"role": msg.role, "content": msg.content})

    try:
        prompt_tokens = len(str(messages)) // 4
        completion_tokens = 0
        
        # Omitimos parameters que algunos modelos de openrouter no aceptan
        extra_args = {}
        if req.temperature > 0:
            extra_args["temperature"] = req.temperature
        
        stream = await client.chat.completions.create(
            model=req.model,
            messages=messages,
            max_tokens=req.max_new_tokens,
            stream=True,
            **extra_args
        )
        
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                completion_tokens += 1
                yield f"data: {json.dumps({'token': chunk.choices[0].delta.content})}\n\n"
                
        yield f"data: {json.dumps({'done': True, 'tokens': {'prompt': prompt_tokens, 'completion': completion_tokens}})}\n\n"
    except Exception as e:
        logger.error(f"Error in stream_openai_compatible_api: {e}")
        msg = str(e)
        if req.provider == "fireworks" and ("model_not_found" in msg.lower() or "does not exist" in msg.lower()):
            msg = (
                "Fireworks devolvio que ese modelo no existe para inferencia. "
                "Puede pasar si el catalogo lista un modelo transitorio o no invocable por chat en este momento. "
                "Probá otro modelo de Fireworks o volvé a buscar."
            )
        if "not a chat model" in msg.lower():
            msg = "❌ Este proveedor indica que el modelo no es para chat. Si es un modelo de imagen, cambiá el proveedor a '☁️ HF API' e intentá de nuevo."
        yield f"data: {json.dumps({'error': msg})}\n\n"


# ──────────────────────────────────────────────────────────────────────────────
# CHAT – Inferencia local con transformers
# ──────────────────────────────────────────────────────────────────────────────

async def stream_local(req: ChatRequest) -> AsyncGenerator[str, None]:
    key = _model_key(req.model)
    with _model_lock:
        model_data = _loaded_models.get(key)
        # Búsqueda por alias corto
        if not model_data:
            for loaded_key, data in _loaded_models.items():
                short_name = _short_model_alias(loaded_key)
                if short_name == key.lower():
                    model_data = data
                    break

    if not model_data:
        yield f"data: {json.dumps({'error': 'El modelo no está cargado. Cargalo primero en la sección Modelos → Local.'})}\n\n"
        return

    try:
        from transformers import TextIteratorStreamer
        import threading
        from PIL import Image

        tokenizer = model_data["tokenizer"]
        processor = model_data.get("processor")
        model = model_data["model"]
        supports_vision = bool(model_data.get("supports_vision"))
        images = []

        if req.images:
            if not supports_vision:
                yield f"data: {json.dumps({'error': 'El modelo local cargado es solo texto y no acepta imágenes.'})}\n\n"
                return
            for img_item in req.images:
                img_bytes = base64.b64decode(img_item.base64)
                images.append(Image.open(BytesIO(img_bytes)).convert("RGB"))

        # Construir prompt con chat template si está disponible
        messages = []
        if req.system_prompt:
            messages.append({"role": "system", "content": req.system_prompt})
        
        user_messages = [m for m in req.messages if m.role == "user"]
        last_user_msg = user_messages[-1] if user_messages else None

        for msg in req.messages:
            is_last_user_with_images = images and msg == last_user_msg
            if is_last_user_with_images:
                content = []
                for _ in images:
                    content.append({"type": "image"})
                content.append({"type": "text", "text": msg.content or "Describí las imágenes."})
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": msg.role, "content": msg.content})

        try:
            template_engine = processor if supports_vision and processor is not None else tokenizer
            input_text = template_engine.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            logger.warning(f"apply_chat_template falló ({e}). Reintentando sin system prompt...")
            # Muchos modelos fallan si se les pasa un "system" prompt. Lo removemos e intentamos de nuevo.
            messages_no_system = [m for m in messages if m["role"] != "system"]
            try:
                input_text = template_engine.apply_chat_template(
                    messages_no_system,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e2:
                logger.error(f"apply_chat_template volvió a fallar ({e2}). Usando fallback crudo.")
                parts = []
                if req.system_prompt:
                    parts.append(f"[SYSTEM] {req.system_prompt}")
                for msg in req.messages:
                    role = "Usuario" if msg.role == "user" else "Asistente"
                    has_img = images and msg == last_user_msg
                    prefix = f"[{len(images)} Imágenes] " if has_img else ""
                    parts.append(f"{role}: {prefix}{msg.content}")
                parts.append("Asistente:")
                input_text = "\n".join(parts)

        if supports_vision and processor is not None and images:
            inputs = processor(
                images=images if len(images) > 1 else images[0],
                text=input_text,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = tokenizer(input_text, return_tensors="pt")

        try:
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            prompt_tokens = inputs["input_ids"].shape[-1]
        except Exception:
            prompt_tokens = len(input_text) // 4
            pass

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": req.max_new_tokens,
            "do_sample": req.temperature > 0,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "repetition_penalty": req.repetition_penalty,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }

        # Ejecutar en hilo para permitir streaming
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        completion_tokens = 0
        for new_text in streamer:
            completion_tokens += 1
            yield f"data: {json.dumps({'token': new_text})}\n\n"
        
        yield f"data: {json.dumps({'done': True, 'tokens': {'prompt': prompt_tokens, 'completion': completion_tokens}})}\n\n"

    except Exception as e:
        logger.error(f"Local inference error: {e}")
        yield f"data: {json.dumps({'error': f'Error de inferencia local: {str(e)}'})}\n\n"


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Streaming SSE: usa modo local o API según `use_local`."""
    if req.use_local and not LOCAL_MODE:
        async def _mode_error():
            yield f"data: {json.dumps({'error': 'Inferencia local deshabilitada. Configurá LLMFRONT_MODE=local para usarla.'})}\n\n"
        return StreamingResponse(
            _mode_error(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )
        
    # Inject RAG context if applicable
    if getattr(rag, 'RAG_ENABLED', False) and getattr(rag, 'collection', None) and rag.collection.count() > 0:
        if req.messages and req.messages[-1].role == "user":
            last_msg = req.messages[-1].content
            if isinstance(last_msg, str):
                context = rag.query_rag_context(last_msg, n_results=3)
                if context:
                    rag_info = f"\n\n[CONTEXTO DE DOCUMENTOS ADJUNTOS]\n{context}\n[FIN DEL CONTEXTO]\n"
                    if req.system_prompt:
                        req.system_prompt += rag_info
                    else:
                        req.system_prompt = f"Eres un asistente que tiene acceso a documentos adjuntos. {rag_info}"
                    
                    # Debug
                    with open("debug_rag.txt", "w", encoding="utf-8") as f:
                        f.write(f"System Prompt: {req.system_prompt}\nUser Msg: {last_msg}")
    
    if req.use_local or req.provider == "local":
        generator = stream_local(req)
    elif req.provider in ["groq", "openrouter", "together", "deepinfra", "fireworks", "baseten"]:
        generator = stream_openai_compatible_api(req)
    else:
        generator = stream_hf_api(req)
        
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@app.post("/rag/upload")
async def upload_rag_document(file: UploadFile = File(...)):
    """Sube un documento y lo procesa para RAG usando ChromaDB."""
    print(f"DEBUG: /rag/upload reached for {file.filename}")
    content = await file.read()
    result = rag.process_and_store_document(file.filename, content)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/rag/clear")
async def clear_rag_index():
    """Limpia el índice de documentos en memoria y disco."""
    if hasattr(rag, 'collection'):
        rag.collection = rag.SimpleBM25()
        if os.path.exists(rag.INDEX_PATH):
            os.remove(rag.INDEX_PATH)
        return {"status": "success", "message": "Índice RAG limpiado."}
    return {"status": "error", "message": "RAG no disponible."}

@app.post("/chat/complete")
async def chat_complete(req: ChatRequest):
    """Respuesta completa (sin streaming) – sólo modo API."""
    token = req.hf_token or HF_TOKEN
    if not token:
        raise HTTPException(status_code=401, detail="HF Token requerido")

    client = InferenceClient(token=token)
    messages = []
    if req.system_prompt:
        messages.append({"role": "system", "content": req.system_prompt})
    for msg in req.messages:
        messages.append({"role": msg.role, "content": msg.content})

    try:
        response = client.chat_completion(
            model=req.model,
            messages=messages,
            max_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stream=False,
        )
        return {"response": response.choices[0].message.content, "model": req.model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if LOCAL_MODE:
    @app.get("/api/version")
    async def ollama_version():
        """Ollama API: Version check (used by Twinny, Continue and other clients as health check)"""
        return {"version": "0.1.32"}

if LOCAL_MODE:
    @app.post("/api/show")
    async def ollama_show(req: dict):
        """Ollama API: Show model info"""
        model_id = req.get("name", req.get("model", ""))
        return {
            "modelfile": "",
            "parameters": "",
            "template": "",
            "details": {"format": "pytorch", "family": "", "parameter_size": "", "quantization_level": ""},
            "model_info": {"general.name": model_id},
        }

if LOCAL_MODE:
    @app.get("/api/tags")
    async def ollama_tags():
        """Ollama API: List downloaded models (Perfect mock)"""
        models = []
        for d in MODELS_CACHE_DIR.iterdir():
            if d.is_dir():
                model_id = d.name.replace("--", "/", 1)
                if _is_downloaded(model_id):
                    # Nombre original
                    models.append({
                        "name": model_id,
                        "model": model_id,
                        "modified_at": datetime.now().isoformat() + "Z",
                        "size": int(_dir_size_gb(d) * 1e9),
                        "digest": "fake-digest-12345",
                        "details": {
                            "parent_model": "",
                            "format": "gguf",
                            "family": "llama",
                            "families": ["llama"],
                            "parameter_size": "1.7B",
                            "quantization_level": "Q4_0"
                        }
                    })
                    # Alias corto (ej: smollm2)
                    short_name = _short_model_alias(model_id)
                    if short_name and short_name != model_id:
                        models.append({
                            "name": short_name,
                            "model": short_name,
                            "modified_at": datetime.now().isoformat() + "Z",
                            "size": int(_dir_size_gb(d) * 1e9),
                            "digest": "fake-digest-12345",
                            "details": {
                                "parent_model": "",
                                "format": "gguf",
                                "family": "llama",
                                "families": ["llama"],
                                "parameter_size": "1.7B",
                                "quantization_level": "Q4_0"
                            }
                        })
        return {"models": models}


async def _ollama_stream(generator, is_chat=True, model_name=""):
    async for chunk in generator:
        if not chunk.startswith("data: "):
            continue
        try:
            data = json.loads(chunk[6:])
        except Exception:
            continue
            
        if "error" in data:
            yield json.dumps({"error": data["error"]}) + "\n"
            return
            
        if "token" in data:
            if is_chat:
                yield json.dumps({
                    "model": model_name,
                    "created_at": datetime.now().isoformat() + "Z",
                    "message": {"role": "assistant", "content": data["token"]},
                    "done": False
                }) + "\n"
            else:
                yield json.dumps({
                    "model": model_name,
                    "created_at": datetime.now().isoformat() + "Z",
                    "response": data["token"],
                    "done": False
                }) + "\n"
                
        if "done" in data and data["done"]:
            if is_chat:
                yield json.dumps({
                    "model": model_name,
                    "created_at": datetime.now().isoformat() + "Z",
                    "message": {"role": "assistant", "content": ""},
                    "done": True
                }) + "\n"
            else:
                yield json.dumps({
                    "model": model_name,
                    "created_at": datetime.now().isoformat() + "Z",
                    "response": "",
                    "done": True
                }) + "\n"

if LOCAL_MODE:
    async def _consume_ollama_stream(generator):
        full_text = ""
        async for chunk in generator:
            if chunk.startswith("data: "):
                try:
                    data = json.loads(chunk[6:])
                    if "token" in data:
                        full_text += data["token"]
                except Exception:
                    pass
        return full_text

    @app.post("/api/chat")
    async def ollama_chat(req: dict):
        """Ollama API: Chat (Robust version)"""
        print(f"\n🚀 OLLAMA CHAT REQUEST: {req}\n")
        model_id = req.get("model", "")
        messages = req.get("messages", [])
        stream   = req.get("stream", True)
        options  = req.get("options", {})

        chat_req = ChatRequest(
            model=model_id,
            messages=[Message(role=m.get("role","user"), content=m.get("content","")) for m in messages],
            use_local=True,
            temperature=options.get("temperature", 0.7),
            top_p=options.get("top_p", 0.9),
            max_new_tokens=options.get("num_predict", 512),
            stream=stream
        )
        generator = stream_local(chat_req)
        
        if stream:
            return StreamingResponse(_ollama_stream(generator, True, model_id), media_type="application/x-ndjson")
        else:
            full_text = await _consume_ollama_stream(generator)
            return {
                "model": model_id,
                "created_at": datetime.now().isoformat() + "Z",
                "message": {"role": "assistant", "content": full_text},
                "done": True
            }

if LOCAL_MODE:
    @app.post("/api/generate")
    async def ollama_generate(req: dict):
        """Ollama API: Generate (Robust version)"""
        print(f"\n🚀 OLLAMA GENERATE REQUEST: {req}\n")
        model_id = req.get("model", "")
        prompt   = req.get("prompt", "")
        system   = req.get("system", "")
        stream   = req.get("stream", True)
        options  = req.get("options", {})

        chat_req = ChatRequest(
            model=model_id,
            messages=[Message(role="user", content=prompt)],
            system_prompt=system,
            use_local=True,
            temperature=options.get("temperature", 0.7),
            top_p=options.get("top_p", 0.9),
            max_new_tokens=options.get("num_predict", 512),
            stream=stream
        )
        generator = stream_local(chat_req)

        if stream:
            return StreamingResponse(_ollama_stream(generator, False, model_id), media_type="application/x-ndjson")
        else:
            full_text = await _consume_ollama_stream(generator)
            return {
                "model": model_id,
                "created_at": datetime.now().isoformat() + "Z",
                "response": full_text,
                "done": True
            }

# ─── OpenAI-compatible endpoint (for Twinny v7+ and other OpenAI clients) ─────
if LOCAL_MODE:
    @app.post("/v1/chat/completions")
    async def openai_chat_completions(req: dict):
        """OpenAI-compatible chat endpoint (used by Twinny v7+ and Continue)"""
        model_id   = req.get("model", "")
        messages   = req.get("messages", [])
        stream     = req.get("stream", True)
        temperature = req.get("temperature", 0.7)
        max_tokens  = req.get("max_tokens", 512)
        top_p       = req.get("top_p", 0.9)

        # Parsear mensajes soportando formato multimodal de OpenAI
        parsed_messages = []
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                # Extraer el texto si viene como lista de diccionarios
                text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                content = "\n".join(text_parts)
            parsed_messages.append(Message(role=m.get("role", "user"), content=str(content)))

        chat_req = ChatRequest(
            model=model_id,
            messages=parsed_messages,
            use_local=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_tokens,
            stream=stream,
        )

        async def openai_stream(generator):
            import time
            async for chunk in generator:
                if not chunk.startswith("data: "):
                    continue
                try:
                    data = json.loads(chunk[6:])
                except Exception:
                    continue
                
                error = data.get("error")
                if error:
                    yield "data: " + json.dumps({
                        "id": "chatcmpl-llmfront",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_id,
                        "choices": [{"delta": {"content": f"⚠️ Error: {error}"}, "index": 0, "finish_reason": "stop"}]
                    }) + "\n\n"
                    yield "data: [DONE]\n\n"
                    continue

                token = data.get("token", "")
                done  = data.get("done", False)
                if token:
                    yield "data: " + json.dumps({
                        "id": "chatcmpl-llmfront",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_id,
                        "choices": [{"delta": {"content": token}, "index": 0, "finish_reason": None}]
                    }) + "\n\n"
                if done:
                    yield "data: " + json.dumps({
                        "id": "chatcmpl-llmfront",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_id,
                        "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]
                    }) + "\n\n"
                    yield "data: [DONE]\n\n"

        if stream:
            return StreamingResponse(openai_stream(stream_local(chat_req)), media_type="text/event-stream")
        else:
            full_text = ""
            async for chunk in stream_local(chat_req):
                if chunk.startswith("data: "):
                    try:
                        data = json.loads(chunk[6:])
                        if "token" in data:
                            full_text += data["token"]
                    except Exception:
                        pass
            import time
            return {
                "id": "chatcmpl-llmfront",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{"message": {"role": "assistant", "content": full_text}, "index": 0, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }

# ─── OpenAI models list (for clients that check /v1/models) ───────────────────
if LOCAL_MODE:
    @app.get("/v1/models")
    async def openai_models():
        """OpenAI-compatible model list"""
        import time
        models = []
        for d in MODELS_CACHE_DIR.iterdir():
            if d.is_dir():
                model_id = d.name.replace("--", "/", 1)
                if _is_downloaded(model_id):
                    models.append({"id": model_id, "object": "model", "created": int(time.time()), "owned_by": "local"})
        return {"object": "list", "data": models}


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/app", response_class=HTMLResponse)
async def serve_frontend():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


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
