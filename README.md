# LLMFront 🤗

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

**LLMFront** is a unified, self-hosted web interface to explore, download, and chat with open-source LLMs — both via the **HuggingFace Inference API (cloud)** and **fully offline on your own hardware** using `transformers`.

---

## 🚀 Features

| Feature | Description |
|---|---|
| ☁️ **Cloud (API) mode** | Chat with any model via HuggingFace Inference API, Groq or OpenRouter |
| 💻 **Local mode** | Download model weights and run inference on your own CPU/GPU |
| 🔀 **Hybrid mode** (`both`) | Use API and local inference simultaneously from the same UI |
| 🔍 **Model Explorer** | Search HuggingFace Hub with filters for task, size (≤3.5B / 4–9B / ≥10B) |
| ✨ **AI-assisted search** | Uses Groq (llama-3.1-8b-instant) to interpret natural language queries |
| 📥 **Download manager** | Download models in the background with real-time progress bar via SSE |
| 🧠 **Load manager** | Load/unload models from RAM/VRAM with live progress feedback |
| 🗑️ **Delete models** | Remove downloaded models from disk directly from the UI |
| ▶️ **Resume downloads** | Partial/incomplete downloads can be resumed without starting over |
| 👁️ **Vision support** | Attach images to chat (multi-image) for vision-capable models |
| 📸 **Camera capture** | Take photos directly from the browser webcam and attach them |
| 📋 **Paste images** | Paste images from clipboard with Ctrl+V |
| ⚡ **Streaming chat** | Token-by-token streaming via Server-Sent Events (SSE) |
| ⚙️ **Full config** | System prompt, temperature, max tokens, top-p, repetition penalty, presets |
| 🌐 **i18n** | UI available in Spanish 🇦🇷 and English 🇺🇸 |
| 🤖 **Ollama-compatible endpoints** | `/api/chat`, `/api/generate`, `/api/tags` — drop-in Ollama mock |
| 🌑 **Glassmorphism dark UI** | Responsive dark-mode interface with smooth animations |

---

## 📦 Installation

**Requirements:**
- Python 3.10+
- NVIDIA GPU with CUDA (recommended for local mode)
- [HuggingFace Token](https://huggingface.co/settings/tokens) (free)
- [Groq API Key](https://console.groq.com/keys) (optional, for AI search)

```bash
# 1. Clone the repo
git clone https://github.com/your-username/llmfront.git
cd llmfront

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
# API-only mode (~100 MB RAM, no torch):
pip install -r requirements-api.txt

# Full local mode (torch + transformers):
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your keys (see below)

# 5. Run
python main.py
```

Open at: `http://localhost:8000`

---

## ⚙️ Configuration (`.env`)

```env
# Operation mode:
#   "api"   → Cloud only. HF Inference API + Groq + OpenRouter. No torch. ~100 MB RAM.
#   "local" → Local only. Downloads + transformers inference. High RAM.
#   "both"  → Hybrid: enables local AND cloud simultaneously.
LLMFRONT_MODE=both

# Enables the "✨ AI Search" button in the model explorer (requires GROQ_API_KEY)
GROQ_AI_SEARCH_ENABLED=true

# HuggingFace token — needed for gated models and API inference
HF_TOKEN=hf_your_token_here

# Groq API key — used for AI-assisted model search and Groq API inference
GROQ_API_KEY=gsk_your_groq_api_key_here
```

---

## 🗂️ Project Structure

```
llmfront/
├── main.py                # FastAPI backend (all endpoints, inference, download manager)
├── requirements.txt       # Full dependencies (torch + transformers)
├── requirements-api.txt   # Lightweight dependencies for API-only mode
├── .env.example           # Environment template
├── run.sh                 # Convenience startup script
└── static/
    ├── index.html         # UI structure and modals
    ├── style.css          # Dark glassmorphism CSS
    ├── app.js             # Frontend state, SSE handling, download/load progress
    └── i18n.js            # Bilingual translation engine
```

---

## 🖥️ Operation Modes

### `LLMFRONT_MODE=api` — Cloud only
- Minimal RAM (~100 MB). No `torch` or `transformers` required.
- Install with `requirements-api.txt`.
- Use the **☁️ API** toggle in the chat header.
- Compatible with HuggingFace Inference API, Groq, and OpenRouter.

### `LLMFRONT_MODE=local` — Local only
- Downloads model weights to `./models_cache/`.
- Loads models into RAM/VRAM using `transformers`.
- Install with `requirements.txt`.
- Use the **💻 Local** toggle in the chat header.

### `LLMFRONT_MODE=both` — Hybrid *(recommended)*
- Enables both modes simultaneously.
- Switch between API and Local mid-conversation from the UI.
- Requires `requirements.txt`.

---

## 🧠 Quantization

When loading models locally, three precision modes are available:

| Mode | VRAM | Quality | Requirement |
|------|------|---------|-------------|
| 🔵 **Full** | High | Best | Any |
| 🟡 **8-bit** | ~50% | Good | CUDA + bitsandbytes |
| 🟢 **4-bit** | ~25% | Acceptable | CUDA + bitsandbytes |

---

## 🤖 Local API Endpoints (Ollama & OpenAI Compatible)

LLMFront exposes local-model endpoints that mimic both the Ollama API and OpenAI API:

| Endpoint | Format | Description |
|---|---|---|
| `GET /api/tags` | Ollama | List downloaded models |
| `POST /api/chat` | Ollama | Chat (streaming or full) |
| `POST /api/generate` | Ollama | Generate (streaming or full) |
| `GET /v1/models` | OpenAI | List downloaded models |
| `POST /v1/chat/completions` | OpenAI | Chat completions (streaming or full) |

These endpoints are only available when `LOCAL_MODE` is enabled (`local` or `both`).

---

## 🧑‍💻 VS Code Integration

Since LLMFront exposes Ollama-compatible endpoints, you can use it as the backend for AI coding assistants in VS Code — no real Ollama installation needed.

### Option 1: [Continue](https://marketplace.visualstudio.com/items?itemName=Continue.continue) *(recommended)*

The most popular open-source AI coding assistant. Supports chat, autocomplete, and context-aware code editing.

1. Install the **Continue** extension from the VS Code Marketplace.
2. Open `~/.continue/config.json` and add LLMFront as a model:

```json
{
  "models": [
    {
      "title": "LLMFront (local)",
      "provider": "ollama",
      "model": "your-model-id/here",
      "apiBase": "http://localhost:8000"
    }
  ]
}
```

3. Replace `"model"` with the exact ID of a model you've downloaded and loaded in LLMFront (visible in `GET /api/tags`).

### Option 2: [Twinny](https://marketplace.visualstudio.com/items?itemName=rjmacarthy.twinny)

Lightweight Copilot alternative optimized for local models. Easier to configure via UI.

1. Install **Twinny** from the VS Code Marketplace.
2. Open Twinny settings (`Ctrl+,` → search "Twinny") and set for both **Chat** and **FIM**:
   - **Provider**: `Ollama`
   - **Hostname**: `localhost` (or `127.0.0.1`)
   - **Port**: `11434` (Ollama's default, now used by LLMFront)
   - **Model name**: Use the alias **`smollm2`** or the full ID from `/api/tags`.

#### 🔧 Troubleshooting Twinny "Connection Error"

If Twinny shows "Connection Error" and you see **no activity** in the Python console when clicking send:

1. **Check Provider**: In Settings (`Ctrl+,`), search for `Twinny: Chat Provider`. It **must** be set to `ollama`.
2. **Reload VS Code**: Press `Ctrl+Shift+P` → `Developer: Reload Window`. This is often required to apply port changes.
3. **Check Firewall**: Ensure your OS isn't blocking VS Code from connecting.

> ⚠️ **The model must be loaded in memory** (not just downloaded). In LLMFront go to **Modelos → Local**, select the model and click **🧠 Cargar en memoria** before using Twinny.

> **Note:** Both extensions require `LLMFRONT_MODE=local` or `both` (so the `/api/*` endpoints are active) and at least one model downloaded and loaded in memory.

---


## 🧩 Recommended Models

### Small (CPU-friendly, ≤ 3.5B)
- `Qwen/Qwen2.5-0.5B-Instruct` — 1 GB, multilingual, excellent for its size
- `Qwen/Qwen2.5-1.5B-Instruct` — 3 GB, strong instruction following
- `HuggingFaceTB/SmolLM2-1.7B-Instruct` — fast, efficient

### Medium (GPU recommended, 4–9B)
- `Qwen/Qwen2.5-7B-Instruct` — 15 GB, multilingual, very capable
- `mistralai/Mistral-7B-Instruct-v0.3` — strong general purpose
- `google/gemma-2-9b-it` — excellent reasoning

### Large (high-end GPU, ≥ 10B)
- `meta-llama/Meta-Llama-3-8B-Instruct` — top open-source quality
- `Qwen/Qwen2.5-14B-Instruct` — multilingual powerhouse

### Vision models (image + text)
- `Qwen/Qwen2-VL-2B-Instruct` — lightweight vision
- `HuggingFaceM4/idefics2-8b` — strong multimodal

---

## 🛠️ Troubleshooting

**Downloads restart the server:**
The watchfiles reloader is configured to ignore `models_cache/` — if it's restarting on downloads, verify `reload_excludes=["models_cache", "venv"]` is present in `main.py`.

**GGUF / tflite / mobile format errors:**
LLMFront uses `transformers` for local inference. It officially supports `.safetensors`, `.bin`, and `.gguf` formats. If you face issues with a specific GGUF model, try downloading the equivalent safetensors format instead.

**Model not available in Inference API:**
Not all HuggingFace models are served by the free Inference API. If you see `❌ Modelo no soportado por la Inference API`, download the model locally or choose a different one.

**Small models give incoherent responses:**
Models under 3B parameters have limited instruction-following ability. Use at least a 1.5B–3B instruct-tuned model for decent quality, or use the API mode with larger cloud models.
