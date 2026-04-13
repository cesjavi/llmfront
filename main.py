"""
LLMFront - Backend FastAPI
Soporta dos modos:
  1. Inference API de HuggingFace (cloud, sin descarga)
  2. Local: descarga el modelo y corre en tu máquina con transformers
"""

import os
import json
import shutil
import asyncio
import logging
from pathlib import Path
from threading import Thread, Lock
from typing import AsyncGenerator, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from huggingface_hub import InferenceClient, snapshot_download
import requests

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN", "")
MODELS_CACHE_DIR = Path("./models_cache")
MODELS_CACHE_DIR.mkdir(exist_ok=True)

app = FastAPI(title="LLMFront", description="HuggingFace LLM Chat Frontend", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Estado global para inferencia local ─────────────────────────────────────

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

class ModelSearchRequest(BaseModel):
    query: str = ""
    task: str = "text-generation"
    size_filter: str = "any"
    limit: int = 20
    hf_token: Optional[str] = None
    use_ai_search: bool = False
    
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
        from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer


        # Detectar device disponible respetando la preferencia del usuario
        use_cuda = torch.cuda.is_available() and device != "cpu"
        if use_cuda:
            target_device = torch.device("cuda")
            dtype = torch.float16
        else:
            target_device = torch.device("cpu")
            dtype = torch.float32

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
            _download_state[key + "_load"]["message"] = "Cargando tokenizer..."
            _download_state[key + "_load"]["progress"] = 25

        # Detección de archivos GGUF
        gguf_files = list(local_dir.rglob("*.gguf"))
        gguf_kwargs = {}
        is_gguf = len(gguf_files) > 0
        if is_gguf:
            selected_gguf = gguf_files[0]
            if quantization == "4bit":
                q4_files = [f for f in gguf_files if "q4" in f.name.lower()]
                if q4_files: selected_gguf = q4_files[0]
            elif quantization == "8bit":
                q8_files = [f for f in gguf_files if "q8" in f.name.lower()]
                if q8_files: selected_gguf = q8_files[0]
            gguf_kwargs["gguf_file"] = selected_gguf.name
            logger.info(f"Detectado modelo GGUF: usando archivo {selected_gguf.name}")

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
            model = AutoModelForCausalLM.from_pretrained(str(local_dir), **model_kwargs)
        except Exception as load_err:
            logger.warning(f"First load attempt failed ({load_err}), retrying without dtype hints...")
            # Fallback: cargar sin optimizaciones (más lento pero compatible con todo)
            model = AutoModelForCausalLM.from_pretrained(
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
                "model": model,
                "model_id": model_id,
                "quantization": quantization,
            }
            _download_state[key + "_load"] = {
                "status": "loaded",
                "progress": 100,
                "message": "✓ Modelo cargado y listo",
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
    return {"status": "ok", "version": "2.0.0", "local_support": True}


@app.get("/system/info")
async def system_info():
    return _system_info()


@app.get("/models/featured")
async def get_featured_models():
    """Modelos curados con info de tamaño y estado local."""
    models = []
    for m in FEATURED_MODELS:
        entry = dict(m)
        entry["is_downloaded"] = _is_downloaded(m["id"])
        entry["is_partial"] = _is_partial(m["id"])
        key = _model_key(m["id"])
        with _model_lock:
            entry["is_loaded"] = key in _loaded_models
        if entry["is_downloaded"]:
            entry["local_size_gb"] = _dir_size_gb(_model_local_dir(m["id"]))
        models.append(entry)
    return {"models": models}


import re
def _guess_size_b(name: str, tags: list) -> float:
    for tag in tags:
        m = re.search(r'(?i)(\d+(?:\.\d+)?)b', tag)
        if m: return float(m.group(1))
    m = re.search(r'(?i)[_-](\d+(?:\.\d+)?)b[_-]', name)
    if m: return float(m.group(1))
    return -1

@app.post("/models/search")
async def search_models(req: ModelSearchRequest):
    token = req.hf_token or HF_TOKEN
    search_query = req.query
    
    # 🧠 Búsqueda Semántica Asistida por IA (Groq)
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
                    "model": "llama3-8b-8192",
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

    try:
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
            
            if req.size_filter != "any":
                size_b = _guess_size_b(mid, tags)
                if size_b > 0:
                    if req.size_filter == "small" and size_b > 3.5: continue
                    if req.size_filter == "medium" and (size_b <= 3.5 or size_b >= 9.5): continue
                    if req.size_filter == "large" and size_b < 9.5: continue

            models.append({
                "id": mid,
                "name": mid,
                "description": (m.get("cardData", {}) or {}).get("description", "") or "",
                "tags": tags[:5],
                "likes": m.get("likes", 0),
                "downloads": m.get("downloads", 0),
                "pipeline_tag": m.get("pipeline_tag", ""),
                "is_downloaded": _is_downloaded(mid),
                "is_partial": _is_partial(mid),
            })
            if len(models) >= req.limit:
                break
        return {"models": models, "total": len(models)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/local")
async def list_local_models():
    """Lista todos los modelos descargados localmente."""
    local = []
    for d in MODELS_CACHE_DIR.iterdir():
        if d.is_dir():
            model_id = d.name.replace("--", "/", 1)
            key = d.name
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
            })
    return {"models": local}


@app.post("/models/download")
async def start_download(req: DownloadRequest):
    """Inicia (o reanuda) la descarga de un modelo en background."""
    key = _model_key(req.model_id)
    with _download_lock:
        existing = _download_state.get(key, {})
        if existing.get("status") == "downloading":
            return {"status": "already_downloading", "message": "Ya se está descargando este modelo"}

    # Detectar si hay una descarga parcial previa para informar
    is_partial = _is_partial(req.model_id)
    size_saved = _dir_size_gb(_model_local_dir(req.model_id)) if is_partial else 0
    if is_partial:
        logger.info(f"Resuming partial download: {req.model_id} ({size_saved} GB saved)")
        with _download_lock:
            _download_state[key] = {
                "status": "downloading",
                "progress": 0,
                "message": f"Reanudando descarga ({size_saved} GB ya guardados)...",
                "model_id": req.model_id,
                "size_saved_gb": size_saved,
            }

    token = req.hf_token or HF_TOKEN
    t = Thread(target=_download_thread, args=(req.model_id, token), daemon=True)
    t.start()
    return {"status": "started" if not is_partial else "resuming", "model_id": req.model_id}


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
            if status in ("done", "error") and load_status in ("loaded", "error", ""):
                if load_status not in ("loading",):
                    break
            timeout += 1.5

        yield f"data: {json.dumps({'_end': True})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


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


@app.post("/models/load")
async def load_model(req: LoadModelRequest):
    """Carga un modelo local en memoria para inferencia."""
    if not _is_downloaded(req.model_id):
        raise HTTPException(status_code=404, detail="El modelo no está descargado")
    
    key = _model_key(req.model_id)
    
    with _model_lock:
        if key in _loaded_models:
            return {"status": "already_loaded", "message": "Modelo ya cargado"}
        # Descargar el modelo anterior si hay uno
        if _loaded_models:
            old_key = next(iter(_loaded_models))
            old_data = _loaded_models.pop(old_key)
            try:
                import torch
                import gc
                del old_data["model"]
                del old_data["tokenizer"]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            except Exception:
                pass
            logger.info(f"Unloaded previous model: {old_key}")

    t = Thread(target=_load_model_thread, args=(req.model_id, req.quantization, req.device), daemon=True)
    t.start()
    return {"status": "loading", "model_id": req.model_id}


@app.post("/models/unload")
async def unload_model(model_id: str):
    """Descarga el modelo de memoria RAM/VRAM."""
    key = _model_key(model_id)
    with _model_lock:
        if key not in _loaded_models:
            return {"status": "not_loaded"}
        data = _loaded_models.pop(key)
    try:
        import torch
        import gc
        if "model" in data:
            del data["model"]
        if "tokenizer" in data:
            del data["tokenizer"]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass
    return {"status": "unloaded", "model_id": model_id}


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
    for msg in req.messages:
        messages.append({"role": msg.role, "content": msg.content})

    try:
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
                yield f"data: {json.dumps({'token': chunk.choices[0].delta.content})}\n\n"
            await asyncio.sleep(0)
        yield f"data: {json.dumps({'done': True})}\n\n"
    except Exception as e:
        msg = str(e)
        if "403" in msg or "401" in msg:
            msg = "Token inválido o sin acceso al modelo."
        elif "404" in msg:
            msg = "Modelo no disponible en Inference API. Intentá descargarlo localmente."
        elif "503" in msg or "loading" in msg.lower():
            msg = "El modelo está cargando en HF. Intentá en unos segundos."
        elif "not supported" in msg.lower():
            msg = "Modelo no soportado por la Inference API. Usá el modo Local."
        yield f"data: {json.dumps({'error': msg})}\n\n"


# ──────────────────────────────────────────────────────────────────────────────
# CHAT – Inferencia local con transformers
# ──────────────────────────────────────────────────────────────────────────────

async def stream_local(req: ChatRequest) -> AsyncGenerator[str, None]:
    key = _model_key(req.model)
    with _model_lock:
        model_data = _loaded_models.get(key)

    if not model_data:
        yield f"data: {json.dumps({'error': 'El modelo no está cargado. Cargalo primero en la sección Modelos → Local.'})}\n\n"
        return

    try:
        from transformers import TextIteratorStreamer
        import threading

        tokenizer = model_data["tokenizer"]
        model = model_data["model"]

        # Construir prompt con chat template si está disponible
        messages = []
        if req.system_prompt:
            messages.append({"role": "system", "content": req.system_prompt})
        for msg in req.messages:
            messages.append({"role": msg.role, "content": msg.content})

        try:
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback: formato simple
            parts = []
            if req.system_prompt:
                parts.append(f"[SYSTEM] {req.system_prompt}")
            for msg in messages:
                role = "Usuario" if msg["role"] == "user" else "Asistente"
                parts.append(f"{role}: {msg['content']}")
            parts.append("Asistente:")
            input_text = "\n".join(parts)

        inputs = tokenizer(input_text, return_tensors="pt")
        # mover al mismo device del modelo
        try:
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except Exception:
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
            "temperature": req.temperature,
            "top_p": req.top_p,
            "repetition_penalty": req.repetition_penalty,
            "do_sample": req.temperature > 0.01,
            "pad_token_id": tokenizer.eos_token_id,
        }

        # Generar en un hilo separado
        gen_thread = threading.Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
        gen_thread.start()

        loop = asyncio.get_event_loop()

        for token_text in streamer:
            yield f"data: {json.dumps({'token': token_text})}\n\n"
            await asyncio.sleep(0)  # ceder control al event loop

        gen_thread.join(timeout=5)
        yield f"data: {json.dumps({'done': True})}\n\n"

    except Exception as e:
        logger.error(f"Local inference error: {e}")
        yield f"data: {json.dumps({'error': f'Error de inferencia local: {str(e)}'})}\n\n"


# ──────────────────────────────────────────────────────────────────────────────
# ENDPOINTS – Chat
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Streaming SSE: usa modo local o API según `use_local`."""
    generator = stream_local(req) if req.use_local else stream_hf_api(req)
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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


import datetime

@app.get("/api/tags")
async def ollama_tags():
    """Ollama API: List downloaded models"""
    models = []
    for d in MODELS_CACHE_DIR.iterdir():
        if d.is_dir():
            model_id = d.name.replace("--", "/", 1)
            if _is_downloaded(model_id):
                models.append({
                    "name": model_id,
                    "model": model_id,
                    "modified_at": datetime.datetime.now().isoformat() + "Z",
                    "size": int(_dir_size_gb(d) * 1e9),
                    "digest": "",
                    "details": {"format": "pytorch", "family": "", "parameter_size": "", "quantization_level": ""}
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
                    "created_at": datetime.datetime.now().isoformat() + "Z",
                    "message": {"role": "assistant", "content": data["token"]},
                    "done": False
                }) + "\n"
            else:
                yield json.dumps({
                    "model": model_name,
                    "created_at": datetime.datetime.now().isoformat() + "Z",
                    "response": data["token"],
                    "done": False
                }) + "\n"
                
        if "done" in data and data["done"]:
            if is_chat:
                yield json.dumps({
                    "model": model_name,
                    "created_at": datetime.datetime.now().isoformat() + "Z",
                    "message": {"role": "assistant", "content": ""},
                    "done": True
                }) + "\n"
            else:
                yield json.dumps({
                    "model": model_name,
                    "created_at": datetime.datetime.now().isoformat() + "Z",
                    "response": "",
                    "done": True
                }) + "\n"

@app.post("/api/chat")
async def ollama_chat(req: OllamaChatRequest):
    """Ollama API: Chat"""
    chat_req = ChatRequest(
        model=req.model,
        messages=[Message(role=m.role, content=m.content) for m in req.messages],
        use_local=True,
        temperature=req.options.get("temperature", 0.7),
        top_p=req.options.get("top_p", 0.9),
        max_new_tokens=req.options.get("num_predict", 512),
        stream=req.stream
    )
    generator = stream_local(chat_req)
    
    if req.stream:
        return StreamingResponse(_ollama_stream(generator, True, req.model), media_type="application/x-ndjson")
    else:
        full_text = ""
        async for chunk in generator:
            if chunk.startswith("data: "):
                try:
                    data = json.loads(chunk[6:])
                    if "token" in data:
                        full_text += data["token"]
                except Exception:
                    pass
        return {
            "model": req.model,
            "created_at": datetime.datetime.now().isoformat() + "Z",
            "message": {"role": "assistant", "content": full_text},
            "done": True
        }

@app.post("/api/generate")
async def ollama_generate(req: OllamaGenerateRequest):
    """Ollama API: Generate"""
    chat_req = ChatRequest(
        model=req.model,
        messages=[Message(role="user", content=req.prompt)],
        system_prompt=req.system,
        use_local=True,
        temperature=req.options.get("temperature", 0.7),
        top_p=req.options.get("top_p", 0.9),
        max_new_tokens=req.options.get("num_predict", 512),
        stream=req.stream
    )
    generator = stream_local(chat_req)

    if req.stream:
        return StreamingResponse(_ollama_stream(generator, False, req.model), media_type="application/x-ndjson")
    else:
        full_text = ""
        async for chunk in generator:
            if chunk.startswith("data: "):
                try:
                    data = json.loads(chunk[6:])
                    if "token" in data:
                        full_text += data["token"]
                except Exception:
                    pass
        return {
            "model": req.model,
            "created_at": datetime.datetime.now().isoformat() + "Z",
            "response": full_text,
            "done": True
        }

# ─── Servir frontend ──────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/app", response_class=HTMLResponse)
async def serve_frontend():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True, reload_excludes=["models_cache", "venv"])
