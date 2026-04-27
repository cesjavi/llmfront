"""
routers/compat.py – Endpoints de compatibilidad Ollama y OpenAI.
Solo se registran si LOCAL_MODE está habilitado.
"""
import json
import time
import logging
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from schemas import Message, ChatRequest
from state import _model_lock, _loaded_models
from local.helpers import MODELS_CACHE_DIR, _model_key, _is_downloaded, _dir_size_gb, _short_model_alias
from inference import stream_local

logger = logging.getLogger("llmfront")
router = APIRouter()


async def _ollama_stream(generator, is_chat=True, model_name=""):
    async for chunk in generator:
        if not chunk.startswith("data: "): continue
        try: data = json.loads(chunk[6:])
        except Exception: continue
        if "error" in data:
            yield json.dumps({"error": data["error"]}) + "\n"
            return
        if "token" in data:
            if is_chat:
                yield json.dumps({"model":model_name,"created_at":datetime.now().isoformat()+"Z",
                                  "message":{"role":"assistant","content":data["token"]},"done":False}) + "\n"
            else:
                yield json.dumps({"model":model_name,"created_at":datetime.now().isoformat()+"Z",
                                  "response":data["token"],"done":False}) + "\n"
        if data.get("done"):
            if is_chat:
                yield json.dumps({"model":model_name,"created_at":datetime.now().isoformat()+"Z",
                                  "message":{"role":"assistant","content":""},"done":True}) + "\n"
            else:
                yield json.dumps({"model":model_name,"created_at":datetime.now().isoformat()+"Z",
                                  "response":"","done":True}) + "\n"


async def _consume_ollama_stream(generator) -> str:
    full = ""
    async for chunk in generator:
        if chunk.startswith("data: "):
            try:
                data = json.loads(chunk[6:])
                if "token" in data: full += data["token"]
            except Exception: pass
    return full


@router.get("/api/version")
async def ollama_version():
    return {"version": "0.1.32"}


@router.post("/api/show")
async def ollama_show(req: dict):
    model_id = req.get("name", req.get("model", ""))
    return {"modelfile":"","parameters":"","template":"",
            "details":{"format":"pytorch","family":"","parameter_size":"","quantization_level":""},
            "model_info":{"general.name": model_id}}


@router.get("/api/tags")
async def ollama_tags():
    models = []
    details = {"parent_model":"","format":"gguf","family":"llama","families":["llama"],"parameter_size":"1.7B","quantization_level":"Q4_0"}
    for d in MODELS_CACHE_DIR.iterdir():
        if not d.is_dir(): continue
        model_id = d.name.replace("--", "/", 1)
        if not _is_downloaded(model_id): continue
        size = int(_dir_size_gb(d) * 1e9)
        base_entry = {"model": model_id, "modified_at": datetime.now().isoformat()+"Z", "size": size, "digest": "fake-digest-12345", "details": details}
        models.append({"name": model_id, **base_entry})
        alias = _short_model_alias(model_id)
        if alias and alias != model_id:
            models.append({"name": alias, **base_entry})
    return {"models": models}


@router.post("/api/chat")
async def ollama_chat(req: dict):
    """Ollama API: Chat"""
    model_id = req.get("model", "")
    stream   = req.get("stream", True)
    options  = req.get("options", {})
    chat_req = ChatRequest(
        model=model_id,
        messages=[Message(role=m.get("role","user"), content=m.get("content","")) for m in req.get("messages",[])],
        use_local=True,
        temperature=options.get("temperature", 0.7),
        top_p=options.get("top_p", 0.9),
        max_new_tokens=options.get("num_predict", 512),
        stream=stream,
    )
    generator = stream_local(chat_req)
    if stream:
        return StreamingResponse(_ollama_stream(generator, True, model_id), media_type="application/x-ndjson")
    full = await _consume_ollama_stream(generator)
    return {"model":model_id,"created_at":datetime.now().isoformat()+"Z","message":{"role":"assistant","content":full},"done":True}


@router.post("/api/generate")
async def ollama_generate(req: dict):
    """Ollama API: Generate"""
    model_id = req.get("model", "")
    stream   = req.get("stream", True)
    options  = req.get("options", {})
    chat_req = ChatRequest(
        model=model_id,
        messages=[Message(role="user", content=req.get("prompt",""))],
        system_prompt=req.get("system",""),
        use_local=True,
        temperature=options.get("temperature", 0.7),
        top_p=options.get("top_p", 0.9),
        max_new_tokens=options.get("num_predict", 512),
        stream=stream,
    )
    generator = stream_local(chat_req)
    if stream:
        return StreamingResponse(_ollama_stream(generator, False, model_id), media_type="application/x-ndjson")
    full = await _consume_ollama_stream(generator)
    return {"model":model_id,"created_at":datetime.now().isoformat()+"Z","response":full,"done":True}


@router.post("/v1/chat/completions")
async def openai_chat_completions(req: dict):
    """OpenAI-compatible chat endpoint (Twinny v7+, Continue)"""
    from inference import stream_local
    model_id    = req.get("model", "")
    messages    = req.get("messages", [])
    stream      = req.get("stream", True)
    temperature = req.get("temperature", 0.7)
    max_tokens  = req.get("max_tokens", 512)
    top_p       = req.get("top_p", 0.9)

    parsed = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):
            content = "\n".join(p.get("text","") for p in content if isinstance(p,dict) and p.get("type")=="text")
        parsed.append(Message(role=m.get("role","user"), content=str(content)))

    chat_req = ChatRequest(model=model_id, messages=parsed, use_local=True,
                           temperature=temperature, top_p=top_p, max_new_tokens=max_tokens, stream=stream)

    async def openai_stream(gen):
        async for chunk in gen:
            if not chunk.startswith("data: "): continue
            try: data = json.loads(chunk[6:])
            except Exception: continue
            error = data.get("error")
            if error:
                yield "data: " + json.dumps({"id":"chatcmpl-llmfront","object":"chat.completion.chunk",
                    "created":int(time.time()),"model":model_id,
                    "choices":[{"delta":{"content":f"Error: {error}"},"index":0,"finish_reason":"stop"}]}) + "\n\n"
                yield "data: [DONE]\n\n"; continue
            token = data.get("token",""); done = data.get("done", False)
            if token:
                yield "data: " + json.dumps({"id":"chatcmpl-llmfront","object":"chat.completion.chunk",
                    "created":int(time.time()),"model":model_id,
                    "choices":[{"delta":{"content":token},"index":0,"finish_reason":None}]}) + "\n\n"
            if done:
                yield "data: " + json.dumps({"id":"chatcmpl-llmfront","object":"chat.completion.chunk",
                    "created":int(time.time()),"model":model_id,
                    "choices":[{"delta":{},"index":0,"finish_reason":"stop"}]}) + "\n\n"
                yield "data: [DONE]\n\n"

    if stream:
        return StreamingResponse(openai_stream(stream_local(chat_req)), media_type="text/event-stream")
    full = ""
    async for chunk in stream_local(chat_req):
        if chunk.startswith("data: "):
            try:
                data = json.loads(chunk[6:])
                if "token" in data: full += data["token"]
            except Exception: pass
    return {"id":"chatcmpl-llmfront","object":"chat.completion","created":int(time.time()),"model":model_id,
            "choices":[{"message":{"role":"assistant","content":full},"index":0,"finish_reason":"stop"}],
            "usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}


@router.get("/v1/models")
async def openai_models():
    """OpenAI-compatible model list"""
    models = []
    for d in MODELS_CACHE_DIR.iterdir():
        if d.is_dir():
            model_id = d.name.replace("--", "/", 1)
            if _is_downloaded(model_id):
                models.append({"id":model_id,"object":"model","created":int(time.time()),"owned_by":"local"})
    return {"object": "list", "data": models}
