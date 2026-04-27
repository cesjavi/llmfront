"""
routers/models.py – Endpoints de búsqueda y metadatos de modelos.
"""
import os
import re
import json
import logging
import httpx
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException

from schemas import ModelSearchRequest, ModelApiCheckRequest
from state import (
    HF_TOKEN, GROQ_API_KEY, TOGETHER_API_KEY, FIREWORKS_API_KEY,
    BASETEN_API_KEY, DEEPINFRA_API_KEY, _model_lock, _loaded_models,
)
from config import GROQ_AI_SEARCH_ENABLED, PROVIDER_BASE_URLS
from local.helpers import (
    _is_downloaded, _is_partial,
    _model_key, _model_local_dir, _dir_size_gb, _read_local_config,
)

logger = logging.getLogger("llmfront")
router = APIRouter()

# ─── Constantes ───────────────────────────────────────────────────────────────

FEATURED_MODELS = [
    {"id": "mistralai/Mistral-7B-Instruct-v0.3", "name": "Mistral 7B Instruct v0.3",
     "description": "Rápido y capaz. Ideal para uso general. ~14 GB.", "tags": ["chat", "instruct", "7B"], "likes": 12000, "size_gb": 14.5},
    {"id": "meta-llama/Meta-Llama-3-8B-Instruct", "name": "Llama 3 8B Instruct",
     "description": "Llama 3 de Meta, muy buena calidad de razonamiento. ~16 GB.", "tags": ["chat", "instruct", "8B", "llama3"], "likes": 25000, "size_gb": 16.0},
    {"id": "google/gemma-2-9b-it", "name": "Gemma 2 9B IT",
     "description": "Modelo de Google con excelente rendimiento. ~18 GB.", "tags": ["chat", "instruct", "9B", "google"], "likes": 8000, "size_gb": 18.0},
    {"id": "Qwen/Qwen2.5-7B-Instruct", "name": "Qwen 2.5 7B Instruct",
     "description": "Multilingüe y con soporte para código. ~15 GB.", "tags": ["chat", "instruct", "7B", "multilingual"], "likes": 9500, "size_gb": 15.2},
    {"id": "microsoft/Phi-3.5-mini-instruct", "name": "Phi-3.5 Mini Instruct",
     "description": "Pequeño pero potente. ~7.6 GB. Ideal para CPU.", "tags": ["chat", "instruct", "3.8B", "microsoft"], "likes": 7000, "size_gb": 7.6},
    {"id": "HuggingFaceH4/zephyr-7b-beta", "name": "Zephyr 7B Beta",
     "description": "Fine-tune de Mistral optimizado para ser útil y honesto. ~14 GB.", "tags": ["chat", "instruct", "7B", "zephyr"], "likes": 6000, "size_gb": 14.0},
    {"id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "name": "TinyLlama 1.1B Chat",
     "description": "Modelo ultrapequeño. ~2.2 GB. Perfecto para probar localmente en CPU.", "tags": ["chat", "1.1B", "tiny", "cpu-friendly"], "likes": 4500, "size_gb": 2.2},
    {"id": "Qwen/Qwen2.5-0.5B-Instruct", "name": "Qwen 2.5 0.5B Instruct",
     "description": "Muy ligero, 0.5B parámetros. ~1 GB. Excelente para CPU.", "tags": ["chat", "instruct", "0.5B", "cpu-friendly"], "likes": 3000, "size_gb": 1.0},
]

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _guess_size_b(name: str, tags: list) -> float:
    for tag in tags:
        m = re.search(r'(?i)(\d+(?:\.\d+)?)b', tag)
        if m: return float(m.group(1))
    m = re.search(r'(?i)[_-](\d+(?:\.\d+)?)b[_-]', name)
    if m: return float(m.group(1))
    return -1

def _extract_model_meta(model_id: str, tags: Optional[list] = None, pipeline_tag: str = "", description: str = "", config: Optional[dict] = None) -> dict:
    tags = tags or []
    config = config or {}
    haystack = " ".join([model_id or "", pipeline_tag or "", description or "",
                         config.get("model_type", "") or "", " ".join(config.get("architectures", []) or []),
                         " ".join(tags)]).lower()
    vision_kw = ("llava","vision","vlm","image-text-to-text","image text to text","idefics","paligemma",
                 "qwen2-vl","qwen-vl","smolvlm","cambrian","bunny","internvl","minicpm-v","glm-4v")
    supports_vision = any(k in haystack for k in vision_kw)
    return {
        "supports_vision": supports_vision,
        "supports_local_chat": True,
        "capability_label": "Vision + Texto" if supports_vision else "Solo texto",
        "model_type": config.get("model_type", ""),
        "architectures": config.get("architectures", []) or [],
        "pipeline_tag": pipeline_tag or "",
    }


def _matches_query(query: str, *values: str) -> bool:
    if not query.strip(): return True
    haystack = " ".join(v for v in values if v).lower()
    return all(t in haystack for t in re.split(r"\s+", query.lower().strip()) if t)

def _passes_size_filter(model_id: str, tags: list, size_filter: str) -> bool:
    if size_filter == "any": return True
    sb = _guess_size_b(model_id, tags)
    if sb <= 0: return True
    if size_filter == "small": return sb <= 3.5
    if size_filter == "medium": return 3.5 < sb < 9.5
    if size_filter == "large": return sb >= 9.5
    return True

def _provider_search_result(model_id, name, description="", tags=None, likes=0, downloads=0, pipeline_tag="", extra_meta=None) -> dict:
    tags = tags or []
    meta = _extract_model_meta(model_id, tags=tags, pipeline_tag=pipeline_tag, description=description, config=extra_meta or {})
    return {"id": model_id, "name": name or model_id, "description": description, "tags": tags[:6],
            "likes": likes, "downloads": downloads, "pipeline_tag": pipeline_tag,
            "is_downloaded": _is_downloaded(model_id), "is_partial": _is_partial(model_id), **meta}

def _process_provider_items(items, query, size_filter, limit, mapper) -> list:
    models = []
    for item in items:
        res = mapper(item)
        if not res: continue
        mid, name, desc, tags, ptag, extra = res
        if not _matches_query(query, mid, name, desc, " ".join(tags)): continue
        if not _passes_size_filter(mid, tags, size_filter): continue
        models.append(_provider_search_result(mid, name, desc, tags, pipeline_tag=ptag, extra_meta=extra))
        if len(models) >= limit: break
    return models

# ─── Búsqueda por proveedor ───────────────────────────────────────────────────

async def _search_openrouter_models(query, size_filter, limit):
    async with httpx.AsyncClient() as c:
        resp = await c.get("https://openrouter.ai/api/v1/models", timeout=20)
        resp.raise_for_status()
        data = resp.json().get("data", [])
    def mapper(item):
        mid = item.get("id", ""); name = item.get("name") or item.get("canonical_slug") or mid
        arch = item.get("architecture", {}) or {}
        tags = [t for t in [arch.get("modality",""), *arch.get("input_modalities",[])] if t]
        return mid, name, item.get("description",""), tags, "conversational", None
    return _process_provider_items(data, query, size_filter, limit, mapper), "openrouter"

async def _search_together_models(query, size_filter, limit, api_key):
    if not api_key: raise HTTPException(400, "Together AI requiere API key.")
    async with httpx.AsyncClient() as c:
        resp = await c.get("https://api.together.xyz/v1/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=20)
        resp.raise_for_status(); data = resp.json()
    def mapper(item):
        mid = item.get("id",""); tags = [item.get("type","")]
        if item.get("organization"): tags.append(item["organization"])
        if item.get("context_length"): tags.append(f"{item['context_length']}ctx")
        return mid, item.get("display_name") or mid, f"Tipo: {item.get('type','unknown')}", tags, "conversational" if item.get("type")=="chat" else "text-generation", None
    models = _process_provider_items(data, query, size_filter, limit, mapper)
    if query and "/" in query and not any(m["id"]==query for m in models):
        try:
            async with httpx.AsyncClient() as c:
                probe = await c.post("https://api.together.xyz/v1/chat/completions",
                    headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"},
                    json={"model":query,"messages":[{"role":"user","content":"Hi"}],"max_tokens":1}, timeout=10)
                if probe.is_success:
                    models.insert(0, _provider_search_result(query, query.split("/")[-1], "Custom Model", ["custom","together"], pipeline_tag="conversational"))
        except Exception: pass
    return models, "together"

async def _is_invocable_fireworks_model(model_id, api_key) -> bool:
    try:
        async with httpx.AsyncClient() as c:
            probe = await c.post("https://api.fireworks.ai/inference/v1/chat/completions",
                headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"},
                json={"model":model_id,"messages":[{"role":"user","content":"Hi"}],"max_tokens":1,"temperature":0.0,"stream":False}, timeout=12)
            return probe.is_success
    except Exception: return False

async def _search_fireworks_models(query, size_filter, limit, api_key):
    if not api_key: raise HTTPException(400, "Fireworks AI requiere API key.")
    async with httpx.AsyncClient() as c:
        resp = await c.get("https://api.fireworks.ai/v1/accounts/fireworks/models",
            headers={"Authorization":f"Bearer {api_key}"},
            params={"filter":"supports_serverless=true","pageSize":min(limit*3,200)}, timeout=20)
        resp.raise_for_status(); data = resp.json().get("models",[])
    async def mapper(item):
        s = item.get("status",{}) or {}
        if s.get("code") and s.get("code")!="OK": return None
        if item.get("supportsServerless") is False: return None
        mid = item.get("name",""); base = item.get("baseModelDetails",{}) or {}
        mt = (base.get("modelType","") or "").lower()
        if any(t in mt for t in ["embed","rerank","audio","vision"]): return None
        tags = [t for t in [base.get("modelType",""), base.get("parameterCount","")] if t]
        if not await _is_invocable_fireworks_model(mid, api_key): return None
        return mid, item.get("displayName") or mid, item.get("description",""), tags, "conversational", {"model_type": base.get("modelType","")}
    models = []
    for item in data:
        res = await mapper(item)
        if not res: continue
        mid, name, desc, tags, ptag, meta = res
        if not _matches_query(query, mid, name, desc, " ".join(tags)): continue
        if not _passes_size_filter(mid, tags, size_filter): continue
        models.append(_provider_search_result(mid, name, desc, tags, pipeline_tag=ptag, extra_meta=meta))
        if len(models) >= limit: break
    if query and "/" in query and not any(m["id"]==query for m in models):
        if await _is_invocable_fireworks_model(query, api_key):
            models.insert(0, _provider_search_result(query, query.split("/")[-1], "Custom Model", ["custom","fireworks"], pipeline_tag="conversational"))
    return models, "fireworks"

async def _search_baseten_models(query, size_filter, limit, api_key):
    if not api_key: raise HTTPException(400, "Baseten requiere API key.")
    async with httpx.AsyncClient() as c:
        resp = await c.get("https://inference.baseten.co/v1/models", headers={"Authorization":f"Api-Key {api_key}"}, timeout=20)
        resp.raise_for_status(); payload = resp.json()
    data = payload.get("data") or payload.get("models") or []
    def mapper(item):
        mid = item.get("id",""); tags = []
        arch = item.get("architecture",{}) or {}
        if arch.get("modality"): tags.append(arch["modality"])
        if item.get("context_length"): tags.append(f"{item['context_length']}ctx")
        return mid, item.get("name") or mid, item.get("description",""), tags, "conversational", None
    models = _process_provider_items(data, query, size_filter, limit, mapper)
    if query and not any(m["id"]==query for m in models):
        try:
            async with httpx.AsyncClient() as c:
                probe = await c.post("https://inference.baseten.co/v1/chat/completions",
                    headers={"Authorization":f"Api-Key {api_key}","Content-Type":"application/json"},
                    json={"model":query,"messages":[{"role":"user","content":"Hi"}],"max_tokens":1}, timeout=10)
                if probe.is_success:
                    models.insert(0, _provider_search_result(query, query, "Custom Model", ["custom","baseten"], pipeline_tag="conversational"))
        except Exception: pass
    return models, "baseten"

async def search_huggingface(query, size_filter, limit, token=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    fetch_limit = limit*4 if size_filter != "any" else limit
    params = {"search": query, "filter": "text-generation", "sort": "likes", "direction": -1, "limit": fetch_limit, "full": False}
    async with httpx.AsyncClient() as c:
        resp = await c.get("https://huggingface.co/api/models", params=params, headers=headers, timeout=10)
        resp.raise_for_status(); data = resp.json()
    def mapper(m):
        mid = m.get("modelId", m.get("id",""))
        return mid, mid.split("/")[-1], f"Author: {m.get('author','unknown')} | Likes: {m.get('likes',0)}", m.get("tags",[]), m.get("pipeline_tag","text-generation"), None
    return _process_provider_items(data, query, size_filter, limit, mapper), "huggingface"

# ─── Disponibilidad ───────────────────────────────────────────────────────────

async def _check_hf_api_availability(model_id, token) -> dict:
    if not token:
        return {"status":"unknown","label":"Sin verificar","mode":None,"detail":"Se necesita un HF Token."}
    headers = {"Authorization":f"Bearer {token}","Content-Type":"application/json"}

    def _classify(resp):
        text = (resp.text or "")[:280]; lo = text.lower()
        if resp.is_success: return "available", text
        if any(t in lo for t in ["not a chat model","model_not_supported","not supported","unsupported","text-generation is not supported"]): return "unsupported", text
        if resp.status_code in (401,403): return "auth", text
        if resp.status_code in (408,409,424,425,429,500,502,503,504): return "transient", text
        return "unknown", text

    chat_status = chat_error = "unknown"
    try:
        async with httpx.AsyncClient() as c:
            resp = await c.post(f"https://api-inference.huggingface.co/models/{model_id}/v1/chat/completions",
                headers=headers, json={"model":model_id,"messages":[{"role":"user","content":"Hola"}],"max_tokens":1,"stream":False,"temperature":0.1}, timeout=20)
            chat_status, chat_error = _classify(resp)
            if chat_status == "available": return {"status":"available","label":"Disponible","mode":"chat","detail":"HF responde por chat completions."}
    except Exception as e: chat_error = str(e)

    text_status = text_error = "unknown"
    try:
        async with httpx.AsyncClient() as c:
            resp = await c.post(f"https://api-inference.huggingface.co/models/{model_id}",
                headers=headers, json={"inputs":"Hola","parameters":{"max_new_tokens":1,"return_full_text":False},"options":{"wait_for_model":False}}, timeout=20)
            text_status, text_error = _classify(resp)
            if text_status == "available": return {"status":"available","label":"Disponible","mode":"text-generation","detail":"HF responde por text-generation."}
    except Exception as e: text_error = str(e)

    if chat_status == "auth" or text_status == "auth":
        return {"status":"unknown","label":"Sin verificar","mode":None,"detail":"Token inválido o sin permisos."}
    if chat_status == "unsupported" and text_status == "unsupported":
        return {"status":"unavailable","label":"No disponible","mode":None,"detail":text_error or chat_error or "HF no expone este modelo."}
    if chat_status == "unsupported":
        return {"status":"available","label":"Disponible","mode":"text-generation","detail":"Funciona por text-generation."}
    if text_status == "unsupported":
        return {"status":"available","label":"Disponible","mode":"chat","detail":"Funciona por chat completions."}
    return {"status":"unknown","label":"Sin verificar","mode":None,"detail":text_error or chat_error or "No se pudo confirmar."}

async def _check_openai_provider_model_availability(provider, model_id, api_key) -> dict:
    base_url = PROVIDER_BASE_URLS.get(provider)
    if not api_key or not base_url:
        return {"status":"unknown","label":"Sin verificar","mode":None,"detail":f"Info insuficiente para {provider}."}
    headers = {"Authorization":f"Bearer {api_key}" if provider!="baseten" else f"Api-Key {api_key}","Content-Type":"application/json"}
    try:
        async with httpx.AsyncClient() as c:
            resp = await c.post(f"{base_url}/chat/completions", headers=headers,
                json={"model":model_id,"messages":[{"role":"user","content":"Hi"}],"max_tokens":1,"temperature":0.0,"stream":False}, timeout=15)
            msg = (resp.text or "")[:280]; lo = msg.lower()
            if resp.is_success: return {"status":"available","label":"Disponible","mode":"chat","detail":f"{provider.upper()} responde."}
            if "model_not_found" in lo or "does not exist" in lo or "not found" in lo:
                return {"status":"unavailable","label":"No disponible","mode":None,"detail":f"{provider.upper()} no reconoce este model id."}
            if "embedding" in lo or "not a chat model" in lo or "not supported" in lo:
                return {"status":"unavailable","label":"No disponible","mode":None,"detail":f"{provider.upper()} indica que el modelo no es para chat."}
            return {"status":"unknown","label":"Sin verificar","mode":None,"detail":msg or "Sin confirmar."}
    except Exception as e:
        return {"status":"unknown","label":"Sin verificar","mode":None,"detail":str(e)}

# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/models/featured")
async def get_featured_models():
    models = []
    for m in FEATURED_MODELS:
        entry = dict(m)
        entry.update(_extract_model_meta(m["id"], tags=m.get("tags",[]), description=m.get("description","")))
        entry["is_downloaded"] = _is_downloaded(m["id"])
        entry["is_partial"] = _is_partial(m["id"])
        key = _model_key(m["id"])
        with _model_lock:
            entry["is_loaded"] = key in _loaded_models
        if entry["is_downloaded"]:
            entry["local_size_gb"] = _dir_size_gb(_model_local_dir(m["id"]))
        models.append(entry)
    return {"models": models}

@router.get("/models/trending")
async def get_trending_models():
    try:
        from huggingface_hub import list_models
        hf_models = list_models(sort="likes", limit=10, filter="text-generation", cardData=True)
        models = []
        for m in hf_models:
            mid = m.modelId
            models.append({
                "id": mid, "name": mid.split('/')[-1].replace('-',' ').title(),
                "description": f"Modelo tendencia en HF. Creado por {mid.split('/')[0]}.",
                "tags": getattr(m,'tags',[])[:4], "likes": getattr(m,'likes',0),
                "is_downloaded": _is_downloaded(mid), "is_partial": _is_partial(mid),
                "is_trending": True, "size_gb": -1,
            })
        return {"models": models}
    except Exception as e:
        logger.error(f"Error fetching trending: {e}")
        return {"models": [], "error": str(e)}

@router.get("/models/size/{model_id:path}")
async def get_model_size(model_id: str):
    try:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
        async with httpx.AsyncClient() as c:
            resp = await c.get(f"https://huggingface.co/api/models/{model_id}", headers=headers, timeout=5)
            if resp.status_code == 200:
                st = resp.json().get("safetensors", {})
                if st.get("total"): return {"size_gb": round(st["total"]/1e9, 2)}
        return {"size_gb": None}
    except Exception:
        return {"size_gb": None}

@router.post("/models/search")
async def search_models_alias(req: ModelSearchRequest):
    """Alias de /search/models para compatibilidad."""
    return await search_models_endpoint(req)

@router.post("/search/models")
async def search_models_endpoint(req: ModelSearchRequest):
    query = req.query.strip()
    provider = (req.provider or "hf").lower()

    if req.use_ai_search and GROQ_AI_SEARCH_ENABLED and query and GROQ_API_KEY:
        try:
            async with httpx.AsyncClient() as c:
                r = await c.post("https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={"model":"llama-3.1-8b-instant","messages":[{"role":"user","content":f"Dame 3 keywords para buscar en HuggingFace: '{query}'. SOLO texto separado por espacios."}],"max_tokens":50},
                    timeout=5)
                if r.is_success:
                    query = r.json()["choices"][0]["message"]["content"].strip().replace('"','').replace("'","")
        except Exception: pass

    api_key = req.api_key or os.getenv(f"{provider.upper()}_API_KEY", "")
    try:
        if provider == "openrouter": models, src = await _search_openrouter_models(query, req.size_filter, req.limit)
        elif provider == "together": models, src = await _search_together_models(query, req.size_filter, req.limit, api_key)
        elif provider == "fireworks": models, src = await _search_fireworks_models(query, req.size_filter, req.limit, api_key)
        elif provider == "baseten": models, src = await _search_baseten_models(query, req.size_filter, req.limit, api_key)
        else: models, src = await search_huggingface(query, req.size_filter, req.limit, req.hf_token or HF_TOKEN)
        return {"models": models, "total": len(models), "source": src}
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {"models": [], "total": 0, "error": str(e)}

@router.post("/models/api-check")
async def check_model_api(req: ModelApiCheckRequest):
    provider = (req.provider or "hf").lower()
    if provider == "hf":
        return await _check_hf_api_availability(req.model_id, req.hf_token or HF_TOKEN)
    api_key = req.api_key or os.getenv(f"{provider.upper()}_API_KEY", "")
    return await _check_openai_provider_model_availability(provider, req.model_id, api_key)
