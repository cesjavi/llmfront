import os
import httpx
import logging
from typing import Optional, Tuple, List, Callable
from fastapi import HTTPException
from state import HF_TOKEN, _is_downloaded, _is_partial

logger = logging.getLogger("llmfront")

# ─── Helpers de Filtrado y Formateo ───────────────────────────────────────────

def _matches_query(query: str, *fields: str) -> bool:
    if not query: return True
    q = query.lower()
    return any(q in (f or "").lower() for f in fields)

def _passes_size_filter(model_id: str, tags: list, size_filter: str) -> bool:
    if size_filter == "any": return True
    
    size_gb = None
    for t in tags:
        if "gb" in t.lower():
            try: size_gb = float(t.lower().replace("gb", ""))
            except: pass
    
    if size_gb is None:
        if "7b" in model_id.lower(): size_gb = 14
        elif "13b" in model_id.lower(): size_gb = 26
        elif "70b" in model_id.lower(): size_gb = 140
        else: return True

    if size_filter == "small": return size_gb < 10
    if size_filter == "medium": return 10 <= size_gb <= 30
    if size_filter == "large": return size_gb > 30
    return True

def _extract_model_meta(model_id: str, tags: list, pipeline_tag: str, description: str = "", config: dict = None) -> dict:
    meta = {"context_length": None, "architecture": None, "modality": "text"}
    lowered_tags = [t.lower() for t in tags]
    for t in tags:
        if "ctx" in t.lower(): meta["context_length"] = t
    if "vision" in lowered_tags or "multimodal" in lowered_tags:
        meta["modality"] = "vision"
    return meta

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
    meta = _extract_model_meta(model_id, tags, pipeline_tag, description, extra_meta)
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

def _process_provider_items(items: list, query: str, size_filter: str, limit: int, mapper: Callable) -> list:
    models = []
    for item in items:
        res = mapper(item)
        if not res: continue
        model_id, name, description, tags, pipeline_tag, extra_meta = res
        if not _matches_query(query, model_id, name, description, " ".join(tags)): continue
        if not _passes_size_filter(model_id, tags, size_filter): continue
        models.append(_provider_search_result(model_id, name, description, tags, pipeline_tag=pipeline_tag, extra_meta=extra_meta))
        if len(models) >= limit: break
    return models

# ─── Proveedores Remotos ──────────────────────────────────────────────────────

async def _search_openrouter_models(query: str, size_filter: str, limit: int) -> Tuple[list, str]:
    async with httpx.AsyncClient() as client:
        resp = await client.get("https://openrouter.ai/api/v1/models", timeout=20)
        resp.raise_for_status()
        data = resp.json().get("data", [])
    def mapper(item):
        mid = item.get("id", "")
        name = item.get("name") or item.get("canonical_slug") or mid
        arch = item.get("architecture", {}) or {}
        tags = [t for t in [arch.get("modality", ""), *arch.get("input_modalities", [])] if t]
        return mid, name, item.get("description", ""), tags, "conversational", None
    return _process_provider_items(data, query, size_filter, limit, mapper), "openrouter"

async def _search_together_models(query: str, size_filter: str, limit: int, api_key: str) -> Tuple[list, str]:
    if not api_key: return [], "together"
    async with httpx.AsyncClient() as client:
        resp = await client.get("https://api.together.xyz/v1/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    def mapper(item):
        mid = item.get("id", "")
        tags = [item.get("type", "")]
        if item.get("organization"): tags.append(item["organization"])
        if item.get("context_length"): tags.append(f"{item['context_length']}ctx")
        return mid, item.get("display_name") or mid, f"Tipo: {item.get('type')}", tags, "conversational" if item.get("type") == "chat" else "text-generation", None
    return _process_provider_items(data, query, size_filter, limit, mapper), "together"

async def _search_fireworks_models(query: str, size_filter: str, limit: int, api_key: str) -> Tuple[list, str]:
    if not api_key: return [], "fireworks"
    async with httpx.AsyncClient() as client:
        resp = await client.get("https://api.fireworks.ai/v1/accounts/fireworks/models", headers={"Authorization": f"Bearer {api_key}"}, params={"filter": "supports_serverless=true"}, timeout=20)
        resp.raise_for_status()
        data = resp.json().get("models", [])
    async def mapper(item):
        status = item.get("status", {}) or {}
        if status.get("code") != "OK" or item.get("supportsServerless") is False: return None
        mid = item.get("name", "")
        base = item.get("baseModelDetails", {}) or {}
        if any(term in (base.get("modelType") or "").lower() for term in ["embed", "rerank", "audio", "vision"]): return None
        tags = [t for t in [base.get("modelType", ""), base.get("parameterCount", "")] if t]
        return mid, item.get("displayName") or mid, item.get("description", ""), tags, "conversational", {"model_type": base.get("modelType", "")}
    models = []
    for item in data:
        res = await mapper(item)
        if not res: continue
        if not _matches_query(query, res[0], res[1], res[2], " ".join(res[3])): continue
        models.append(_provider_search_result(*res[:5], extra_meta=res[5]))
        if len(models) >= limit: break
    return models, "fireworks"

async def _search_baseten_models(query: str, size_filter: str, limit: int, api_key: str) -> Tuple[list, str]:
    if not api_key: return [], "baseten"
    async with httpx.AsyncClient() as client:
        resp = await client.get("https://inference.baseten.co/v1/models", headers={"Authorization": f"Api-Key {api_key}"}, timeout=20)
        resp.raise_for_status()
        data = resp.json().get("data", []) or resp.json().get("models", [])
    def mapper(item):
        mid = item.get("id", "")
        tags = []
        arch = item.get("architecture", {}) or {}
        if arch.get("modality"): tags.append(arch["modality"])
        return mid, item.get("name") or mid, item.get("description", ""), tags, "conversational", None
    return _process_provider_items(data, query, size_filter, limit, mapper), "baseten"

async def search_huggingface(query: str, size_filter: str, limit: int, token: str = None) -> Tuple[list, str]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    fetch_limit = limit * 4 if size_filter != "any" else limit
    params = {"search": query, "filter": "text-generation", "sort": "likes", "direction": -1, "limit": fetch_limit, "full": False}
    async with httpx.AsyncClient() as client:
        resp = await client.get("https://huggingface.co/api/models", params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    def mapper(m):
        mid = m.get("modelId", m.get("id", ""))
        name = mid.split("/")[-1]
        tags = m.get("tags", [])
        desc = f"Author: {m.get('author', 'unknown')} | Likes: {m.get('likes', 0)}"
        return mid, name, desc, tags, m.get("pipeline_tag", "text-generation"), None
    return _process_provider_items(data, query, size_filter, limit, mapper), "huggingface"

# ─── Disponibilidad ──────────────────────────────────────────────────────────

async def _check_hf_api_availability(model_id: str, token: str) -> dict:
    if not token: return {"status": "unknown", "label": "Sin verificar", "detail": "Token requerido"}
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"https://api-inference.huggingface.co/models/{model_id}/v1/chat/completions", headers=headers, json={"model": model_id, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1}, timeout=15)
            if resp.is_success: return {"status": "available", "label": "Disponible", "mode": "chat", "detail": "HF responde."}
    except: pass
    return {"status": "unknown", "label": "Sin verificar", "detail": "No se pudo confirmar"}

async def _check_openai_provider_model_availability(provider: str, model_id: str, api_key: str) -> dict:
    base_urls = {"groq": "https://api.groq.com/openai/v1", "openrouter": "https://openrouter.ai/api/v1", "together": "https://api.together.xyz/v1", "fireworks": "https://api.fireworks.ai/inference/v1", "baseten": "https://inference.baseten.co/v1"}
    url = base_urls.get(provider)
    if not api_key or not url: return {"status": "unknown", "label": "Sin verificar", "detail": "Info insuficiente"}
    headers = {"Authorization": f"Bearer {api_key}" if provider != "baseten" else f"Api-Key {api_key}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{url}/chat/completions", headers=headers, json={"model": model_id, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1}, timeout=15)
            if resp.is_success: return {"status": "available", "label": "Disponible", "mode": "chat", "detail": f"{provider.upper()} responde."}
            return {"status": "unavailable", "label": "No disponible", "detail": resp.text[:200]}
    except Exception as e: return {"status": "unknown", "label": "Sin verificar", "detail": str(e)}
