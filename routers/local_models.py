"""
routers/local_models.py – Endpoints de gestión de modelos locales (download, load, unload).
Solo se registran si LOCAL_MODE está habilitado.
"""
import asyncio
import json
import logging
import shutil
from threading import Thread

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from schemas import DownloadRequest, LoadModelRequest
from state import (
    HF_TOKEN, _model_lock, _download_lock, _download_state, _loaded_models,
)
from local.helpers import (
    MODELS_CACHE_DIR, _model_key, _model_local_dir, _dir_size_gb,
    _is_downloaded, _is_partial, _release_loaded_resources,
    _short_model_alias, _read_local_config, _system_info,
)
from local.downloader import DownloadCancelled, _async_raise, _download_thread
from local.loader import _load_model_thread
from routers.models import _extract_model_meta

logger = logging.getLogger("llmfront")
router = APIRouter()


@router.get("/models/local")
async def list_local_models():
    """Lista todos los modelos descargados localmente."""
    local = []
    for d in MODELS_CACHE_DIR.iterdir():
        if not d.is_dir(): continue
        model_id = d.name.replace("--", "/", 1)
        key = d.name
        config = _read_local_config(d)
        meta = _extract_model_meta(model_id, config=config)
        with _model_lock:
            is_loaded = key in _loaded_models
        size = _dir_size_gb(d)
        is_complete = _is_downloaded(model_id)
        with _download_lock:
            dl_state = _download_state.get(key, {})
            is_loading_now = dl_state.get("status") == "downloading"
        local.append({
            "id": model_id, "name": model_id.split("/")[-1],
            "local_dir": str(d), "size_gb": size,
            "is_loaded": is_loaded, "is_complete": is_complete,
            "is_partial": not is_complete and any(d.iterdir()),
            "is_downloading": is_loading_now,
            **meta,
        })
    return {"models": local}


@router.post("/models/download")
async def start_download(req: DownloadRequest):
    """Inicia (o reanuda) la descarga de un modelo en background."""
    key = _model_key(req.model_id)
    with _download_lock:
        if _download_state.get(key, {}).get("status") == "downloading":
            return {"status": "already_downloading", "message": "Ya se está descargando este modelo"}

    is_partial = _is_partial(req.model_id)
    size_saved = _dir_size_gb(_model_local_dir(req.model_id)) if is_partial else 0
    with _download_lock:
        _download_state[key] = {
            "status": "downloading", "progress": 0,
            "message": f"Reanudando descarga ({size_saved} GB ya guardados)..." if is_partial else "Iniciando descarga...",
            "model_id": req.model_id, "thread_id": None,
        }
        if is_partial: _download_state[key]["size_saved_gb"] = size_saved

    token = req.hf_token or HF_TOKEN

    def run_thread():
        import threading
        with _download_lock:
            _download_state[key]["thread_id"] = threading.get_ident()
        _download_thread(req.model_id, token, _model_key, _model_local_dir, _dir_size_gb)

    Thread(target=run_thread, daemon=True).start()
    return {"status": "started" if not is_partial else "resuming", "model_id": req.model_id}


@router.post("/models/download/cancel")
async def cancel_download(model_id: str):
    key = _model_key(model_id)
    with _download_lock:
        state = _download_state.get(key)
        if state and state.get("status") == "downloading" and "thread_id" in state:
            try:
                _async_raise(state["thread_id"], DownloadCancelled)
                state["status"] = "cancelled"
                state["message"] = "Descarga cancelada"
                return {"status": "cancelled"}
            except Exception as e:
                return {"status": "error", "message": str(e)}
    return {"status": "not_downloading"}


@router.get("/models/download/{model_id:path}/progress")
async def download_progress_sse(model_id: str):
    """SSE stream del progreso de descarga y carga."""
    async def generate():
        key = _model_key(model_id)
        load_key = key + "_load"
        prev = None
        timeout = 0
        while timeout < 3600:
            await asyncio.sleep(1.5)
            with _download_lock: dl = dict(_download_state.get(key, {}))
            with _model_lock: ld = dict(_download_state.get(load_key, {}))
            local_dir = _model_local_dir(model_id)
            if local_dir.exists() and dl.get("status") == "downloading":
                dl["size_downloaded_gb"] = _dir_size_gb(local_dir)
            combined = {"download": dl, "load": ld}
            if combined != prev:
                yield f"data: {json.dumps(combined)}\n\n"
                prev = combined
            status = dl.get("status"); load_status = ld.get("status", "")
            if status in ("done","error","cancelled") and load_status not in ("loading",):
                break
            timeout += 1.5
        yield f"data: {json.dumps({'_end': True})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","Connection":"keep-alive","X-Accel-Buffering":"no"})


@router.get("/models/download/{model_id:path}/status")
async def download_status(model_id: str):
    key = _model_key(model_id)
    with _download_lock: dl = dict(_download_state.get(key, {"status": "idle"}))
    with _model_lock:
        ld = dict(_download_state.get(key + "_load", {}))
        is_loaded = key in _loaded_models
    local_dir = _model_local_dir(model_id)
    if local_dir.exists() and dl.get("status") == "downloading":
        dl["size_downloaded_gb"] = _dir_size_gb(local_dir)
    return {"download": dl, "load": ld, "is_downloaded": _is_downloaded(model_id), "is_loaded": is_loaded}


@router.post("/models/load")
async def load_model(req: LoadModelRequest):
    """Carga un modelo local en memoria para inferencia."""
    if not _is_downloaded(req.model_id):
        raise HTTPException(404, "El modelo no está descargado")
    key = _model_key(req.model_id)
    with _model_lock:
        if key in _loaded_models:
            return {"status": "already_loaded", "message": "Modelo ya cargado"}
        if _loaded_models:
            old_key = next(iter(_loaded_models))
            _release_loaded_resources(_loaded_models.pop(old_key))
            logger.info(f"Unloaded previous model: {old_key}")

    Thread(target=_load_model_thread, args=(
        req.model_id, req.quantization, req.device,
        _model_key, _model_local_dir, _dir_size_gb, _is_partial,
        _extract_model_meta, _system_info,
    ), daemon=True).start()
    return {"status": "loading", "model_id": req.model_id}


@router.post("/models/unload")
async def unload_model(model_id: str):
    key = _model_key(model_id)
    with _model_lock:
        if key not in _loaded_models:
            return {"status": "not_loaded"}
        data = _loaded_models.pop(key)
    _release_loaded_resources(data)
    return {"status": "unloaded", "model_id": model_id}


@router.delete("/models/local/{model_id:path}")
async def delete_local_model(model_id: str):
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
