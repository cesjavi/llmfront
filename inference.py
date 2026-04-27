"""
inference.py – Motores de inferencia: HF API, OpenAI-compatible y local (transformers).
"""
import os
import json
import asyncio
import logging
import base64
import httpx
from io import BytesIO
from typing import AsyncGenerator, Optional

from huggingface_hub import InferenceClient
from openai import AsyncOpenAI

from schemas import ChatRequest, Message
from state import (
    HF_TOKEN, GROQ_API_KEY, TOGETHER_API_KEY, FIREWORKS_API_KEY,
    BASETEN_API_KEY, DEEPINFRA_API_KEY, _model_lock, _loaded_models,
)
from config import PROVIDER_BASE_URLS

logger = logging.getLogger("llmfront")


# ─── Helpers de mensajes ──────────────────────────────────────────────────────

def _prepare_messages(req: ChatRequest) -> list:
    """Prepara el historial inyectando imágenes en el último mensaje del usuario."""
    messages = []
    if req.system_prompt:
        messages.append({"role": "system", "content": req.system_prompt})
    user_msgs = [m for m in req.messages if m.role == "user"]
    last_user_msg = user_msgs[-1] if user_msgs else None
    for msg in req.messages:
        if msg == last_user_msg and req.images:
            content = [{"type":"image_url","image_url":{"url":f"data:{img.mime_type};base64,{img.base64}"}} for img in req.images]
            content.append({"type": "text", "text": msg.content or "Describe las imágenes."})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": msg.role, "content": msg.content})
    return messages


def _plain_text_prompt(messages: list, system_prompt: Optional[str] = None) -> str:
    parts = []
    if system_prompt: parts.append(f"[SYSTEM]\n{system_prompt}")
    for msg in messages:
        role = getattr(msg, "role", msg.get("role","user")) if isinstance(msg, (dict, Message)) else "user"
        content = getattr(msg, "content", msg.get("content","")) if isinstance(msg, (dict, Message)) else str(msg)
        parts.append(f"[{role.upper()}]\n{content}")
    parts.append("[ASSISTANT]\n")
    return "\n\n".join(parts)


def _extract_hf_chunk_text(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices: return ""
    choice = choices[0] or {}
    delta = choice.get("delta") or {}
    message = choice.get("message") or {}
    return delta.get("content") or choice.get("text") or message.get("content") or ""


# ─── HF Inference API ────────────────────────────────────────────────────────

async def stream_hf_api(req: ChatRequest) -> AsyncGenerator[str, None]:
    token = req.hf_token or HF_TOKEN
    if not token:
        yield f"data: {json.dumps({'error': 'Se necesita un HF Token para usar la Inference API.'})}\n\n"
        return

    client = InferenceClient(token=token)
    messages = _prepare_messages(req)

    async def _try_chat_completions_raw(comp_ref: list):
        try:
            async with httpx.AsyncClient() as c:
                async with c.stream("POST",
                    f"https://api-inference.huggingface.co/models/{req.model}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                    json={"model":req.model,"messages":messages,"max_tokens":req.max_new_tokens,
                          "temperature":req.temperature,"top_p":req.top_p,"stream":True}, timeout=60) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if not line or not line.startswith("data: "): continue
                        payload = line[6:]
                        if payload == "[DONE]": break
                        txt = _extract_hf_chunk_text(json.loads(payload))
                        if txt:
                            comp_ref[0] += 1
                            yield f"data: {json.dumps({'token': txt})}\n\n"
            return True
        except Exception as e:
            logger.error(f"Raw HF chat fallback failed: {e}")
            return False

    async def _try_text_generation(comp_ref: list):
        try:
            prompt = _plain_text_prompt(req.messages, req.system_prompt)
            stream = client.text_generation(prompt, model=req.model, max_new_tokens=req.max_new_tokens,
                                            temperature=req.temperature, top_p=req.top_p, stream=True)
            for chunk in stream:
                token_text = chunk if isinstance(chunk, str) else getattr(chunk, "token", None)
                if hasattr(token_text, "text"): token_text = token_text.text
                if token_text:
                    comp_ref[0] += 1
                    yield f"data: {json.dumps({'token': token_text})}\n\n"
                await asyncio.sleep(0)
            return True
        except Exception as e:
            logger.error(f"HF text_generation fallback failed: {e}")
            return False

    async def _try_image_generation():
        try:
            prompt = ""
            for m in reversed(messages):
                if m["role"] == "user":
                    if isinstance(m["content"], list):
                        for p in m["content"]:
                            if isinstance(p, dict) and p.get("type") == "text": prompt = p.get("text",""); break
                    else: prompt = str(m["content"])
                    break
            if not prompt: prompt = "A random image"
            img = client.text_to_image(prompt, model=req.model)
            buf = BytesIO(); img.save(buf, format="JPEG")
            img_str = base64.b64encode(buf.getvalue()).decode()
            yield f"data: {json.dumps({'token': f'![Generada por {req.model}](data:image/jpeg;base64,{img_str})' + chr(10)})}\n\n"
            return True
        except Exception as e:
            logger.error(f"HF image fallback failed: {e}")
            return False

    try:
        prompt_tokens = len(str(messages)) // 4
        comp_ref = [0]
        try:
            stream = client.chat_completion(model=req.model, messages=messages, max_tokens=req.max_new_tokens,
                                            temperature=req.temperature, top_p=req.top_p, stream=True)
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    comp_ref[0] += 1
                    yield f"data: {json.dumps({'token': chunk.choices[0].delta.content})}\n\n"
                await asyncio.sleep(0)
            yield f"data: {json.dumps({'done': True, 'tokens': {'prompt': prompt_tokens, 'completion': comp_ref[0]}})}\n\n"
            return
        except Exception as e:
            msg = str(e).lower()
            if "not a chat model" not in msg and "model_not_supported" not in msg and "conversational" not in msg: raise e

        raw_success = False
        async for chunk in _try_chat_completions_raw(comp_ref):
            yield chunk; raw_success = True
        if raw_success:
            yield f"data: {json.dumps({'done': True, 'tokens': {'prompt': prompt_tokens, 'completion': comp_ref[0]}})}\n\n"
            return

        if not req.images:
            gen_success = False
            async for chunk in _try_text_generation(comp_ref):
                yield chunk; gen_success = True
            if gen_success:
                yield f"data: {json.dumps({'done': True, 'tokens': {'prompt': prompt_tokens, 'completion': comp_ref[0]}})}\n\n"
                return

        img_success = False
        async for chunk in _try_image_generation():
            yield chunk; img_success = True
        if img_success:
            yield f"data: {json.dumps({'done': True, 'tokens': {'prompt': 0, 'completion': 0}})}\n\n"
            return

        raise Exception("Todos los métodos de inferencia para este modelo fallaron.")
    except Exception as e:
        logger.error(f"Final error in stream_hf_api: {e}")
        msg = str(e)
        if "403" in msg or "401" in msg: msg = "Token inválido o sin acceso al modelo."
        elif "404" in msg: msg = "Modelo no disponible en Inference API. Intentá descargarlo localmente."
        elif "503" in msg or "loading" in msg.lower(): msg = "El modelo está cargando en HF. Intentá en unos segundos."
        yield f"data: {json.dumps({'error': msg})}\n\n"


# ─── OpenAI-compatible providers ─────────────────────────────────────────────

async def stream_openai_compatible_api(req: ChatRequest) -> AsyncGenerator[str, None]:
    env_keys = {"groq": GROQ_API_KEY, "openrouter": os.getenv("OPENROUTER_API_KEY",""),
                "together": TOGETHER_API_KEY, "deepinfra": DEEPINFRA_API_KEY,
                "fireworks": FIREWORKS_API_KEY, "baseten": BASETEN_API_KEY}
    key = req.api_key or env_keys.get(req.provider, "")
    url = PROVIDER_BASE_URLS.get(req.provider)
    if not key or not url:
        yield f"data: {json.dumps({'error': f'API Key o URL no configurada para {req.provider}.'})}\n\n"
        return
    client = AsyncOpenAI(api_key=key, base_url=url)
    messages = _prepare_messages(req)
    try:
        prompt_tokens = len(str(messages)) // 4
        completion_tokens = 0
        extra = {}
        if req.temperature > 0: extra["temperature"] = req.temperature
        stream = await client.chat.completions.create(model=req.model, messages=messages,
                                                      max_tokens=req.max_new_tokens, stream=True, **extra)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                completion_tokens += 1
                yield f"data: {json.dumps({'token': chunk.choices[0].delta.content})}\n\n"
        yield f"data: {json.dumps({'done': True, 'tokens': {'prompt': prompt_tokens, 'completion': completion_tokens}})}\n\n"
    except Exception as e:
        logger.error(f"Error in stream_openai_compatible_api: {e}")
        msg = str(e)
        if req.provider == "fireworks" and ("model_not_found" in msg.lower() or "does not exist" in msg.lower()):
            msg = "Fireworks indica que ese modelo no existe para inferencia. Probá otro modelo."
        if "not a chat model" in msg.lower():
            msg = "Este proveedor indica que el modelo no es para chat."
        yield f"data: {json.dumps({'error': msg})}\n\n"


# ─── Inferencia local (transformers) ─────────────────────────────────────────

async def stream_local(req: ChatRequest) -> AsyncGenerator[str, None]:
    from local.helpers import _model_key, _short_model_alias
    key = _model_key(req.model)
    with _model_lock:
        model_data = _loaded_models.get(key)
        if not model_data:
            for lk, data in _loaded_models.items():
                if _short_model_alias(lk) == key.lower():
                    model_data = data; break

    if not model_data:
        yield f"data: {json.dumps({'error': 'El modelo no está cargado. Cargalo primero en Modelos → Local.'})}\n\n"
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
                yield f"data: {json.dumps({'error': 'El modelo local es solo texto y no acepta imágenes.'})}\n\n"
                return
            for img_item in req.images:
                images.append(Image.open(BytesIO(base64.b64decode(img_item.base64))).convert("RGB"))

        messages = []
        if req.system_prompt: messages.append({"role":"system","content":req.system_prompt})
        user_messages = [m for m in req.messages if m.role == "user"]
        last_user_msg = user_messages[-1] if user_messages else None

        for msg in req.messages:
            if images and msg == last_user_msg:
                content = [{"type":"image"} for _ in images]
                content.append({"type":"text","text":msg.content or "Describí las imágenes."})
                messages.append({"role":"user","content":content})
            else:
                messages.append({"role":msg.role,"content":msg.content})

        engine = processor if supports_vision and processor else tokenizer
        try:
            input_text = engine.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            logger.warning(f"apply_chat_template falló ({e}). Reintentando sin system prompt...")
            try:
                input_text = engine.apply_chat_template([m for m in messages if m["role"]!="system"], tokenize=False, add_generation_prompt=True)
            except Exception as e2:
                logger.error(f"apply_chat_template volvió a fallar ({e2}). Usando fallback crudo.")
                parts = [f"[SYSTEM] {req.system_prompt}"] if req.system_prompt else []
                for msg in req.messages:
                    role = "Usuario" if msg.role=="user" else "Asistente"
                    parts.append(f"{role}: {msg.content}")
                parts.append("Asistente:")
                input_text = "\n".join(parts)

        if supports_vision and processor and images:
            inputs = processor(images=images[0] if len(images)==1 else images, text=input_text, padding=True, return_tensors="pt")
        else:
            inputs = tokenizer(input_text, return_tensors="pt")

        try:
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k,v in inputs.items()}
            prompt_tokens = inputs["input_ids"].shape[-1]
        except Exception:
            prompt_tokens = len(input_text) // 4

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = {**inputs, "streamer":streamer, "max_new_tokens":req.max_new_tokens,
                      "do_sample":req.temperature>0, "temperature":req.temperature,
                      "top_p":req.top_p, "repetition_penalty":req.repetition_penalty,
                      "pad_token_id":tokenizer.pad_token_id or tokenizer.eos_token_id}

        threading.Thread(target=model.generate, kwargs=gen_kwargs).start()
        completion_tokens = 0
        for new_text in streamer:
            completion_tokens += 1
            yield f"data: {json.dumps({'token': new_text})}\n\n"
        yield f"data: {json.dumps({'done': True, 'tokens': {'prompt': prompt_tokens, 'completion': completion_tokens}})}\n\n"

    except Exception as e:
        logger.error(f"Local inference error: {e}")
        yield f"data: {json.dumps({'error': f'Error de inferencia local: {str(e)}'})}\n\n"
