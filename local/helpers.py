"""
local/helpers.py – Funciones utilitarias compartidas sobre el sistema de archivos local.
"""
import os
import json
import logging
from pathlib import Path
from typing import Optional

from state import _model_lock, _download_lock, _download_state, _loaded_models, MODELS_DIR

logger = logging.getLogger("llmfront")

MODELS_CACHE_DIR = Path(MODELS_DIR)


def _model_key(model_id: str) -> str:
    return model_id.replace("/", "--")


def _model_local_dir(model_id: str) -> Path:
    return MODELS_CACHE_DIR / _model_key(model_id)


def _dir_size_gb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / 1e9, 2)


def _is_downloaded(model_id: str) -> bool:
    """Verifica que el directorio tiene TODOS los pesos completos del modelo."""
    d = _model_local_dir(model_id)
    if not d.exists(): return False
    if any(d.rglob("*.lock")): return False
    if any(d.rglob("*.incomplete")): return False
    index_file = d / "model.safetensors.index.json"
    if index_file.exists():
        try:
            idx = json.loads(index_file.read_text())
            required = set(idx.get("weight_map", {}).values())
            for shard in required:
                if not (d / shard).exists(): return False
            return len(required) > 0
        except Exception:
            pass
    weight_patterns = ["*.safetensors", "*.bin", "*.gguf", "*.pt", "*.litertlm", "*.tflite", "*.onnx"]
    return any(any(d.rglob(p)) for p in weight_patterns)


def _is_partial(model_id: str) -> bool:
    """Directorio existe pero descarga incompleta."""
    d = _model_local_dir(model_id)
    return d.exists() and not _is_downloaded(model_id)


def _get_loaded_model_key() -> Optional[str]:
    with _model_lock:
        if _loaded_models:
            return next(iter(_loaded_models))
    return None


def _short_model_alias(model_id_or_key: str) -> str:
    normalized = model_id_or_key.replace("--", "/")
    return normalized.split("/")[-1].split("-")[0].lower()


def _release_loaded_resources(data: dict) -> None:
    try:
        import gc, torch
        for k in ("model", "tokenizer", "processor"):
            if k in data: del data[k]
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass


def _system_info() -> dict:
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
            info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except ImportError:
        info["cuda_available"] = False
    return info


def _read_local_config(local_dir: Path) -> dict:
    cfg = local_dir / "config.json"
    if not cfg.exists(): return {}
    try: return json.loads(cfg.read_text(encoding="utf-8"))
    except Exception: return {}
