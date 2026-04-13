# LLMFront 🤗

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

**LLMFront** is a unified, self-hosted web interface that allows you to explore, download, and chat with Artificial Intelligence models (LLMs) both in the **cloud (via API)** and **locally offline** using your own CPU and GPU (native compatibility with Hugging Face `transformers`).

## 🚀 Key Features

- 🔍 **Model Explorer (API/Download)** – Search through thousands of models in the HF Hub via the Inference API or download them directly.
- 💻 **Native Local Execution** – Download weights to disk and load them into RAM/VRAM in the background.
- 🔌 **Smart Hardware Filter** – Filter web searches based on model sizes (Parameters: Millions/Billions) directly from the native UI (e.g., ≤ 3.5B for low RAM, >10B for high-end rigs).
- ⚡ **Quantization Assistant (Memory Check)** – A dynamic interface that determines if you need full precision, 8-bit, or 4-bit loading based on your hardware before triggering an Out of Memory error.
- 🌐 **Internationalization (i18n)** – Automatically translated UI and Base Context (System Prompt) according to your browser's language setting (🇦🇷 Spanish or 🇺🇸 English).
- 💬 **Streaming Web Chat** – Fluid, uninterrupted streaming responses processed via Server-Sent Events (SSE).
- ⚙️ **Full Configuration** – Freely edit Base Prompts, temperature, max tokens, top p, and repetition penalty.
- 🤖 **Ollama Endpoints Mock** – Direct compatibility on `/api/generate`, `/api/chat`, and `/api/tags` routes that emulate native Ollama endpoints.
- 🌑 **Glassmorphism UI** – Immersive dark visuals that are highly responsive on both mobile and PC.

## 📦 Requirements and Installation

- Python 3.10+
- (Recommended) NVIDIA GPU with CUDA support for efficient local generation.
- Free Hugging Face Token (available at https://huggingface.co/settings/tokens)
- (Optional) Free Groq API Key for AI Semantic Search capability (https://console.groq.com/keys)

### Step-by-Step Installation

```bash
# 1. Clone and Enter Directory
git clone https://github.com/your-username/llmfront.git
cd llmfront

# 2. Required: Virtual Environment setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Setting Up API Keys (.env)
# We provide a template file for you to copy:
cp .env.example .env

# Edit .env and paste your master tokens:
# HF_TOKEN=hf_...
# GROQ_API_KEY=gsk_...

# 4. Start Uvicorn Master Server
python main.py
```
Open it in your browser (or on your phone within your local network) at: `http://localhost:8000`

## 🛠️ Code Structure

```text
llmfront/
├── main.py           # Heavy FastAPI backend (Ollama mock + HuggingFace Local/Cloud manager)
├── requirements.txt  # Python Libraries
├── .gitignore        # Exclusions (prevents pushing heavy models_cache to your root)
└── static/
    ├── index.html    # Core UI and DOM structure
    ├── style.css     # Dark Neon-glass rendering CSS
    ├── app.js        # JavaScript State Control engine
    └── i18n.js       # Dynamic translation engine for bilingual menus
```

## 🧠 Common Troubleshooting

- **Python server restarts and kills downloads:** Make sure you are running `main.py` as strictly packaged. It includes a patch telling the *auto-reloader* to ignore massive HF-JSON background file drops to avoid infinite restart loops.
- **LiteRT / .tflite / GGUF issues:** Getting an "unable to load tokenizer" error means you forced raw Python `transformers` to load a mobile or C++ binary format file. Try to filter and download only standard `.safetensors` variants when working strictly inside this frontend logic.

## 🤝 Recommended Models (Free to try on Cloud/Local)
- `mistralai/Mistral-7B-Instruct-v0.3`
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
