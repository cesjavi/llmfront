"""
local/downloader.py – Hilo de descarga de modelos desde HuggingFace Hub.
"""
import ctypes
import logging
from pathlib import Path
from huggingface_hub import snapshot_download

from state import _download_lock, _download_state

logger = logging.getLogger("llmfront")


class DownloadCancelled(Exception):
    pass


def _async_raise(tid, exctype):
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def _download_thread(model_id: str, token: str, model_key_fn, model_local_dir_fn, dir_size_gb_fn):
    """Descarga un modelo en background. Acepta helpers como argumentos para evitar imports circulares."""
    key = model_key_fn(model_id)
    local_dir = model_local_dir_fn(model_id)

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

        size = dir_size_gb_fn(local_dir)
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
    except Exception as e:
        logger.error(f"Download error [{model_id}]: {e}")
        # NO borrar archivos parciales — huggingface_hub los reutiliza para reanudar.
        size_saved = dir_size_gb_fn(local_dir) if local_dir.exists() else 0
        with _download_lock:
            _download_state[key] = {
                "status": "partial",
                "progress": 0,
                "message": f"Descarga interrumpida ({size_saved} GB guardados). Podés reintentar para continuar.",
                "model_id": model_id,
                "size_saved_gb": size_saved,
            }
