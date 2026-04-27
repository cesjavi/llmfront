# LLMFront — Especificación Técnica

> Versión: 0.9 | Python 3.11+ | FastAPI

Backend de chat para modelos de lenguaje. Soporta inferencia local con Transformers/PyTorch y múltiples proveedores cloud (HuggingFace, Groq, OpenRouter, Together, DeepInfra, Fireworks, Baseten). Expone una API compatible con Ollama y OpenAI para integrarse con clientes como Twinny, Continue, o cualquier frontend custom.

---

## 1. Modos de Operación

Configurados mediante `LLMFRONT_MODE` en `.env`:

| Modo | Descripción | Dependencias extras |
|------|-------------|-------------------|
| `api` | Solo cloud. Sin torch. ~100 MB RAM. | Ninguna |
| `local` | Descarga y ejecuta modelos localmente con Transformers. | torch, transformers, Pillow, accelerate |
| `both` | Híbrido: cloud + inferencia local simultánea. | Ídem a `local` |

Los routers `/models/local`, `/models/download/*`, `/models/load`, `/models/unload`, `/api/*` (Ollama) y `/v1/*` (OpenAI) **solo se registran si `LOCAL_MODE = True`**.

---

## 2. Estructura de Archivos

```
llmfront/
├── main.py                  # Orquestador: crea app, registra middlewares y routers
├── config.py                # Constantes globales (modo, URLs, proveedores)
├── schemas.py               # Modelos Pydantic de request/response
├── state.py                 # Estado global compartido (locks, dicts, API keys)
├── inference.py             # Motores de inferencia: HF API, OpenAI-compat, local
├── rag.py                   # RAG con BM25 puro (sin dependencias vectoriales externas)
│
├── local/
│   ├── __init__.py
│   ├── helpers.py           # Filesystem: _is_downloaded, _model_local_dir, _system_info, etc.
│   ├── downloader.py        # Hilo de descarga via huggingface_hub.snapshot_download
│   └── loader.py            # Hilo de carga con AutoModelForCausalLM / AutoModelForImageTextToText
│
├── routers/
│   ├── __init__.py
│   ├── models.py            # GET/POST /models/* — búsqueda, featured, trending, api-check
│   ├── local_models.py      # LOCAL_MODE: /models/local, /models/download/*, /models/load
│   ├── chat.py              # POST /chat/stream, /chat/complete, /rag/*
│   └── compat.py            # LOCAL_MODE: /api/* (Ollama), /v1/* (OpenAI)
│
├── static/
│   └── index.html           # Frontend SPA (HTML + JS vanilla)
│
├── requirements.txt         # Dependencias completas (local + cloud)
├── requirements-api.txt     # Solo dependencias cloud (sin torch)
├── .env.example
└── run.sh
```

---

## 3. Variables de Entorno (`.env`)

```env
# Modo de operación: "api", "local" o "both"
LLMFRONT_MODE=api

# Activa búsqueda AI con Groq en el explorador de modelos
GROQ_AI_SEARCH_ENABLED=true

# Directorio de modelos locales (default: ./models_cache)
MODELS_DIR=./models_cache

# Host y puerto del servidor
HOST=127.0.0.1
PORT=8000

# API Keys
HF_TOKEN=hf_...
GROQ_API_KEY=gsk_...
OPENROUTER_API_KEY=sk-or-...
TOGETHER_API_KEY=...
DEEPINFRA_API_KEY=...
FIREWORKS_API_KEY=...
BASETEN_API_KEY=...
```

---

## 4. Dependencias

### Siempre requeridas
```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
huggingface_hub>=0.22.0
sse-starlette>=1.8.2
python-dotenv>=1.0.0
httpx>=0.27.0
openai>=1.0.0
pypdf>=4.0.0
```

### Solo en modo `local` / `both`
```
transformers @ git+https://github.com/huggingface/transformers.git
torch>=2.4.0
Pillow>=10.0.0
accelerate>=1.1.1
bitsandbytes>=0.44.10   # Cuantización 4/8bit (requiere CUDA)
psutil>=5.9.0
protobuf>=3.20.0
sentencepiece>=0.1.99
tiktoken>=0.7.0
gguf>=0.6.0
```

---

## 5. Módulos del Backend

### `config.py`
Constantes de configuración global. Se importa en todos los módulos.

```python
APP_VERSION: str                    # "0.9"
LLMFRONT_MODE: str                  # "api" | "local" | "both"
LOCAL_MODE: bool                    # True si LLMFRONT_MODE in ("local", "both")
GROQ_AI_SEARCH_ENABLED: bool
CLOUD_PROVIDERS: set[str]           # {"groq", "openrouter", "together", ...}
PROVIDER_BASE_URLS: dict[str, str]  # URL base por proveedor OpenAI-compat
```

### `state.py`
Estado global compartido entre módulos. Sin lógica de negocio.

```python
_model_lock: threading.Lock       # protege _loaded_models
_download_lock: threading.Lock    # protege _download_state

_loaded_models: Dict[str, dict]   # model_key → {model, tokenizer, processor, ...}
_download_state: Dict[str, dict]  # model_key → {status, progress, message, ...}

HF_TOKEN, GROQ_API_KEY, TOGETHER_API_KEY,
FIREWORKS_API_KEY, BASETEN_API_KEY, DEEPINFRA_API_KEY: str

MODELS_DIR: str   # os.getenv("MODELS_DIR", "./models_cache")
```

### `schemas.py`
Modelos Pydantic.

```python
class Message:
    role: str
    content: str

class ImageItem:
    base64: str
    mime_type: str

class ChatRequest:
    model: str
    messages: List[Message]
    system_prompt: Optional[str] = "You are a helpful assistant."
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    stream: bool = True
    hf_token: Optional[str] = None
    use_local: bool = False
    provider: str = "hf"   # "hf"|"groq"|"openrouter"|"together"|"deepinfra"|"fireworks"|"baseten"|"local"
    api_key: Optional[str] = None
    images: Optional[List[ImageItem]] = None

class ModelSearchRequest:
    query: str = ""
    task: str = "text-generation"
    size_filter: str = "any"   # "any" | "small" (<=3.5B) | "medium" (3.5-9.5B) | "large" (>=9.5B)
    limit: int = 20
    hf_token: Optional[str] = None
    provider: str = "hf"
    api_key: Optional[str] = None
    use_ai_search: bool = False

class ModelApiCheckRequest:
    model_id: str
    provider: str = "hf"
    api_key: Optional[str] = None
    hf_token: Optional[str] = None

class DownloadRequest:
    model_id: str
    hf_token: Optional[str] = None
    quantization: str = "none"

class LoadModelRequest:
    model_id: str
    quantization: str = "none"   # "none" | "4bit" | "8bit"
    device: str = "auto"          # "auto" | "cpu" | "cuda"
```

### `local/helpers.py`
Filesystem y hardware. Sin FastAPI, importable desde cualquier módulo.

```python
MODELS_CACHE_DIR: Path

_model_key(model_id) -> str           # "author/model" -> "author--model"
_model_local_dir(model_id) -> Path    # MODELS_CACHE_DIR / _model_key(model_id)
_dir_size_gb(path) -> float
_is_downloaded(model_id) -> bool
    # Verifica: directorio existe, sin .lock/.incomplete,
    # si hay model.safetensors.index.json -> valida shards individuales,
    # sino -> cualquiera de *.safetensors/*.bin/*.gguf/*.pt/*.litertlm
_is_partial(model_id) -> bool         # Directorio existe pero _is_downloaded() es False
_get_loaded_model_key() -> Optional[str]
_short_model_alias(model_id) -> str   # "HuggingFaceTB/SmolLM2-1.7B" -> "smollm2"
_release_loaded_resources(data) -> None  # del model/tokenizer/processor + cuda.empty_cache()
_system_info() -> dict               # {cpu_count, ram_total_gb, ram_available_gb, cuda_available, vram_gb, ...}
_read_local_config(local_dir) -> dict # Lee config.json del modelo
```

### `local/downloader.py`
Descarga en background thread via `huggingface_hub.snapshot_download`.

```python
class DownloadCancelled(Exception): pass

_async_raise(tid, exctype)
    # Inyecta excepción en un thread para cancelación segura

_download_thread(model_id, token, model_key_fn, model_local_dir_fn, dir_size_gb_fn)
    # Actualiza _download_state[key] con {status, progress, message}
    # ignore_patterns: *.msgpack, *.h5, flax_model*, tf_model*, rust_model*
    # En error: deja archivos parciales (snapshot_download los reanuda)
    # En DownloadCancelled: status="cancelled"
```

### `local/loader.py`
Carga modelos en memoria. Background thread.

```python
_load_model_thread(model_id, quantization, device,
                   model_key_fn, model_local_dir_fn, dir_size_gb_fn,
                   is_partial_fn, extract_meta_fn, system_info_fn)
    # 1. Rechaza si _is_partial (descarga incompleta)
    # 2. Valida RAM/VRAM disponible:
    #    - 4bit: ~30% del tamaño en disco
    #    - 8bit: ~55% del tamaño en disco
    #    - none: 100% (float16 GPU / float32 CPU)
    # 3. Detecta GGUF -> selecciona archivo optimo según quantization
    # 4. Detecta vision -> usa AutoModelForImageTextToText + AutoProcessor
    # 5. Detecta LiteRT (.litertlm) -> error descriptivo
    # 6. Carga con BitsAndBytesConfig si 4bit/8bit (requiere CUDA)
    # 7. Guarda en _loaded_models[key] = {model, tokenizer, processor, ...}
    # 8. Actualiza _download_state[key+"_load"] = {status, progress, message}
```

### `inference.py`
Tres motores de inferencia. Todos devuelven `AsyncGenerator[str, None]`.

**Formato SSE de cada chunk:**
```
data: {"token": "texto"}\n\n
data: {"done": true, "tokens": {"prompt": N, "completion": M}}\n\n
data: {"error": "mensaje"}\n\n
```

```python
async def stream_hf_api(req: ChatRequest)
    # Cascada de fallbacks:
    # 1. InferenceClient.chat_completion() (sync loop con asyncio.sleep(0))
    # 2. POST /models/{id}/v1/chat/completions via httpx async stream
    # 3. InferenceClient.text_generation() (si no hay imagenes)
    # 4. InferenceClient.text_to_image() -> devuelve markdown con imagen base64

async def stream_openai_compatible_api(req: ChatRequest)
    # AsyncOpenAI(api_key=..., base_url=PROVIDER_BASE_URLS[req.provider])
    # omite temperature=0 para compatibilidad con ciertos proveedores

async def stream_local(req: ChatRequest)
    # 1. Busca en _loaded_models por key exacto, luego por alias corto
    # 2. Construye prompt con apply_chat_template (3 fallbacks):
    #    a. engine.apply_chat_template(messages, ...)
    #    b. sin system prompt si falla
    #    c. concatenacion manual [SYSTEM]/[USER]/[ASSISTANT]
    # 3. Si hay imagenes: AutoProcessor(images=..., text=...)
    # 4. TextIteratorStreamer en Thread separado
```

### `rag.py`
RAG con BM25 puro (Python stdlib). Sin ChromaDB, sin embeddings.

```python
class SimpleBM25:          # k1=1.5, b=0.75
    add_documents(docs, metadatas)
    search(query, top_k=3) -> list[(doc, meta, score)]
    count() -> int

# Al importar el módulo: load_index() se llama automáticamente
load_index()               # Deserializa rag_index.pkl
save_index()               # Serializa con pickle

process_and_store_document(filename, content_bytes) -> dict
    # Formatos: PDF (pypdf), UTF-8, latin-1
    # Chunking: 1500 chars, overlap 300
    # Persiste en rag_index.pkl

query_rag_context(query, n_results=3) -> str
    # Devuelve fragmentos "--- Fragmento de X ---\ntexto"
    # Se inyecta en system_prompt antes de enviar al LLM
```

---

## 6. API Endpoints

### Globales (todos los modos)

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Redirect 307 a `/app` |
| GET | `/health` | `{status, version, mode, local_support}` |
| GET | `/healthz` | `{status, app, ollama_compat: true}` |
| GET | `/config` | Features habilitadas y proveedores disponibles |
| GET | `/system/info` | RAM, VRAM, CPU disponibles |
| GET | `/models/featured` | Lista curada de 8 modelos recomendados |
| GET | `/models/trending` | Top 10 trending de HF Hub |
| GET | `/models/size/{model_id:path}` | Tamaño en GB desde safetensors HF API |
| POST | `/models/search` | Alias de `/search/models` |
| POST | `/search/models` | Búsqueda multi-proveedor con filtros |
| POST | `/models/api-check` | Verifica invocabilidad via API |
| POST | `/chat/stream` | SSE streaming de chat |
| POST | `/chat/complete` | Respuesta completa (solo HF API) |
| POST | `/rag/upload` | Sube documento para RAG |
| POST | `/rag/clear` | Limpia índice RAG |
| GET | `/app` | Sirve el SPA frontend |

### Solo en `LOCAL_MODE`

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/models/local` | Lista modelos descargados con metadata |
| POST | `/models/download` | Inicia/reanuda descarga |
| POST | `/models/download/cancel` | Cancela descarga (inyecta DownloadCancelled) |
| GET | `/models/download/{id:path}/progress` | SSE: `{download:{...}, load:{...}}` |
| GET | `/models/download/{id:path}/status` | Estado puntual sin SSE |
| POST | `/models/load` | Carga modelo en RAM/VRAM |
| POST | `/models/unload` | Libera modelo de memoria |
| DELETE | `/models/local/{id:path}` | Elimina del disco (rechaza si cargado) |
| GET | `/api/version` | Ollama: `{"version": "0.1.32"}` |
| POST | `/api/show` | Ollama: info de modelo |
| GET | `/api/tags` | Ollama: lista modelos + aliases cortos |
| POST | `/api/chat` | Ollama: chat (stream NDJSON) |
| POST | `/api/generate` | Ollama: generate (stream NDJSON) |
| POST | `/v1/chat/completions` | OpenAI-compat: chat (SSE con `data: [DONE]`) |
| GET | `/v1/models` | OpenAI-compat: lista modelos locales |

---

## 7. Flujos Principales

### Descarga de Modelo
```
POST /models/download
  -> verifica que no esté descargando ya
  -> Thread(_download_thread)
       -> snapshot_download(repo_id, local_dir, ignore_patterns=[...])
       -> _download_state[key] = {status, progress, message}
  -> cliente: GET /models/download/{id}/progress  (SSE, polling 1.5s)
       -> {download: {...}, load: {...}}
```

### Carga de Modelo
```
POST /models/load
  -> _is_downloaded() o 404
  -> si hay modelo previo: _release_loaded_resources() + pop
  -> Thread(_load_model_thread)
       -> valida RAM/VRAM
       -> detecta GGUF / vision / LiteRT
       -> carga tokenizer/processor + modelo
       -> BitsAndBytesConfig si 4bit/8bit
       -> _loaded_models[key] = {model, tokenizer, processor, ...}
       -> _download_state[key+"_load"] = {status, progress}
```

### Chat Streaming
```
POST /chat/stream
  -> si RAG activo: query_rag_context() -> inyecta en system_prompt
  -> dispatch por provider/use_local:
       local/use_local=True  -> stream_local()
       CLOUD_PROVIDERS       -> stream_openai_compatible_api()
       default (hf)          -> stream_hf_api()
  -> StreamingResponse(generator, "text/event-stream")
```

### Búsqueda de Modelos
```
POST /search/models
  -> si use_ai_search + GROQ_API_KEY:
       reformula query con Groq llama-3.1-8b-instant
  -> por provider:
       "hf"         -> GET huggingface.co/api/models?filter=text-generation
       "openrouter" -> GET openrouter.ai/api/v1/models
       "together"   -> GET api.together.xyz/v1/models
       "fireworks"  -> GET api.fireworks.ai/v1/accounts/fireworks/models
                        + probe de invocabilidad por modelo
       "baseten"    -> GET inference.baseten.co/v1/models
  -> _matches_query() + _passes_size_filter()
  -> enriquece con _is_downloaded(), _is_partial(), _extract_model_meta()
```

---

## 8. Sistema RAG

- **Motor:** BM25 puro (Python stdlib, sin ChromaDB, sin embeddings)
- **Persistencia:** `rag_index.pkl` (pickle)
- **Formatos:** PDF (pypdf), texto plano (UTF-8 / latin-1)
- **Chunking:** 1500 chars, overlap 300 chars
- **Inyección:** contexto se agrega al `system_prompt` antes del LLM
- **Limpieza:** `POST /rag/clear` resetea `SimpleBM25()` y borra el .pkl

---

## 9. Compatibilidad Ollama y OpenAI

### Ollama (`/api/*`)
- `GET /api/version` -> `{"version": "0.1.32"}`
- `GET /api/tags` -> modelos descargados + alias sin prefijo `author/`
- `POST /api/chat` -> stream NDJSON `{"model","created_at","message":{"role","content"},"done"}`
- `POST /api/generate` -> stream NDJSON `{"model","created_at","response","done"}`
- Body esperado: `{model, messages/prompt, stream, options: {temperature, top_p, num_predict}}`

### OpenAI (`/v1/*`)
- `POST /v1/chat/completions` -> parsea `content` como string O lista de dicts (multimodal)
- Stream: `{"id","object","choices":[{"delta":{"content":"..."},"finish_reason":null}]}`
- Final: `{"choices":[{"delta":{},"finish_reason":"stop"}]}` + `data: [DONE]`
- `GET /v1/models` -> `{"object":"list","data":[{"id","object":"model","owned_by":"local"}]}`

---

## 10. Detección de Modelos Vision

Por keywords en model_id, tags, pipeline_tag o `config.json.model_type`:
```
llava, vision, vlm, image-text-to-text, idefics, paligemma,
qwen2-vl, qwen-vl, smolvlm, cambrian, bunny, internvl, minicpm-v, glm-4v
```

Si `supports_vision=True`:
- Loader usa `AutoModelForImageTextToText` y `AutoProcessor`
- Chat acepta `ImageItem` en `req.images` (base64 + mime_type)
- Prompt incluye `{"type": "image"}` entries antes del texto

---

## 11. Modelos Curados (Featured)

Hardcodeados en `routers/models.py::FEATURED_MODELS`:

| ID | Tamaño |
|----|--------|
| mistralai/Mistral-7B-Instruct-v0.3 | 14.5 GB |
| meta-llama/Meta-Llama-3-8B-Instruct | 16.0 GB |
| google/gemma-2-9b-it | 18.0 GB |
| Qwen/Qwen2.5-7B-Instruct | 15.2 GB |
| microsoft/Phi-3.5-mini-instruct | 7.6 GB |
| HuggingFaceH4/zephyr-7b-beta | 14.0 GB |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 2.2 GB |
| Qwen/Qwen2.5-0.5B-Instruct | 1.0 GB |

---

## 12. Cuantización (modo local)

| Tipo | Requisito | Factor de RAM/VRAM |
|------|-----------|-------------------|
| `none` | CPU (float32) o CUDA (float16) | 1.0x tamaño |
| `8bit` | CUDA + bitsandbytes | ~0.55x |
| `4bit` | CUDA + bitsandbytes | ~0.30x |
| GGUF | Nativo en transformers | Pre-cuantizado |

El loader valida disponibilidad antes de intentar cargar y lanza `RuntimeError` descriptivo.

---

## 13. Filtros de Tamaño en Búsqueda

Inferido desde nombre del modelo con regex `(\d+(?:\.\d+)?)b`:

| Filtro | Rango |
|--------|-------|
| `small` | <= 3.5B |
| `medium` | 3.5B – 9.5B |
| `large` | >= 9.5B |
| `any` | Sin filtro |

---

## 14. Arranque

```bash
# API mode (sin torch, ~100 MB RAM)
pip install -r requirements-api.txt
LLMFRONT_MODE=api python main.py

# Local mode (con torch)
pip install -r requirements.txt
LLMFRONT_MODE=local python main.py

# Uvicorn directo
uvicorn main:app --host 0.0.0.0 --port 8000 --reload \
  --reload-exclude models_cache --reload-exclude venv
```

El `__main__` de `main.py` abre el browser en `http://HOST:PORT/app` tras 1.5 segundos.

---

## 15. Convenciones Internas

- **model_key**: `"author/model-name"` -> `"author--model-name"` (slash reemplazado por doble guión)
- **_download_state keys**: `model_key` para descarga, `model_key + "_load"` para carga
- **SSE newlines**: `\n\n` reales (no `\\n\\n` literales)
- **Imports**: todas las constantes de config al top del módulo, nunca dentro de funciones
- **Thread safety**: siempre usar `_model_lock` para `_loaded_models` y `_download_lock` para `_download_state`
- **_is_downloaded / _is_partial**: solo importar desde `local.helpers`, nunca desde `state`
