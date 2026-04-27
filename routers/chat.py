"""
routers/chat.py – Endpoints de chat (/chat/stream, /chat/complete, /rag/*).
"""
import os
import json
import logging

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from huggingface_hub import InferenceClient

from schemas import ChatRequest
from state import HF_TOKEN
from config import LOCAL_MODE
from inference import stream_hf_api, stream_openai_compatible_api, stream_local

logger = logging.getLogger("llmfront")
router = APIRouter()

CLOUD_PROVIDERS = {"groq", "openrouter", "together", "deepinfra", "fireworks", "baseten"}


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Streaming SSE: usa modo local, proveedor cloud o HF API."""
    from config import LOCAL_MODE
    from inference import stream_hf_api, stream_openai_compatible_api, stream_local

    if req.use_local and not LOCAL_MODE:
        async def _mode_error():
            yield f"data: {json.dumps({'error': 'Inferencia local deshabilitada. Configurá LLMFRONT_MODE=local.'})}\n\n"
        return StreamingResponse(_mode_error(), media_type="text/event-stream",
                                 headers={"Cache-Control":"no-cache","Connection":"keep-alive","X-Accel-Buffering":"no"})

    # Inject RAG context
    import rag
    if getattr(rag, "RAG_ENABLED", False) and getattr(rag, "collection", None) and rag.collection.count() > 0:
        if req.messages and req.messages[-1].role == "user":
            last_msg = req.messages[-1].content
            if isinstance(last_msg, str):
                context = rag.query_rag_context(last_msg, n_results=3)
                if context:
                    rag_info = f"\n\n[CONTEXTO DE DOCUMENTOS ADJUNTOS]\n{context}\n[FIN DEL CONTEXTO]\n"
                    req.system_prompt = (req.system_prompt or "Eres un asistente.") + rag_info

    if req.use_local or req.provider == "local":
        generator = stream_local(req)
    elif req.provider in CLOUD_PROVIDERS:
        generator = stream_openai_compatible_api(req)
    else:
        generator = stream_hf_api(req)

    return StreamingResponse(generator, media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","Connection":"keep-alive","X-Accel-Buffering":"no"})


@router.post("/chat/complete")
async def chat_complete(req: ChatRequest):
    """Respuesta completa (sin streaming) – solo modo API."""
    token = req.hf_token or HF_TOKEN
    if not token:
        raise HTTPException(401, "HF Token requerido")
    client = InferenceClient(token=token)
    messages = []
    if req.system_prompt:
        messages.append({"role": "system", "content": req.system_prompt})
    for msg in req.messages:
        messages.append({"role": msg.role, "content": msg.content})
    try:
        response = client.chat_completion(model=req.model, messages=messages,
                                          max_tokens=req.max_new_tokens, temperature=req.temperature,
                                          top_p=req.top_p, stream=False)
        return {"response": response.choices[0].message.content, "model": req.model}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/rag/upload")
async def upload_rag_document(file: UploadFile = File(...)):
    """Sube un documento y lo procesa para RAG."""
    import rag
    content = await file.read()
    result = rag.process_and_store_document(file.filename, content)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@router.post("/rag/clear")
async def clear_rag_index():
    """Limpia el índice de documentos en memoria y disco."""
    import rag
    if hasattr(rag, "collection"):
        rag.collection = rag.SimpleBM25()
        if os.path.exists(rag.INDEX_PATH):
            os.remove(rag.INDEX_PATH)
        return {"status": "success", "message": "Índice RAG limpiado."}
    return {"status": "error", "message": "RAG no disponible."}
