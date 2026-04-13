# LLMFront 🤗

Chat con cualquier modelo de **Hugging Face** desde una interfaz web moderna y configurable.

## Características

- 🔍 **Explorador de modelos** – Busca entre miles de modelos de HF Hub
- ⭐ **Modelos destacados** – Selección curada de los mejores modelos de chat
- 💬 **Chat con streaming** – Respuestas en tiempo real token a token via SSE
- ⚙️ **Configuración completa** – System prompt, temperatura, top-p, max tokens, etc.
- 🎛️ **Presets rápidos** – Preciso, Balanceado, Creativo, Código
- 💾 **Persistencia local** – Token y configuración se guardan en el browser
- 📤 **Exportar chat** – Descarga la conversación en Markdown
- 🌑 **Dark mode** – Diseño glassmorphism con acento HuggingFace

## Requisitos

- Python 3.8+
- Token de Hugging Face (gratis en https://huggingface.co/settings/tokens)

## Instalación y uso

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. (Opcional) Configurar token
cp .env.example .env
# Editar .env y poner tu HF_TOKEN

# 3. Iniciar el servidor
python main.py
# o con el script:
./run.sh

# 4. Abrir en el browser
# http://localhost:8000/app
```

## Estructura

```
llmfront/
├── main.py           # Backend FastAPI
├── requirements.txt  # Dependencias Python
├── run.sh            # Script de inicio
├── .env.example      # Variables de entorno
└── static/
    ├── index.html    # Frontend principal
    ├── style.css     # Estilos dark mode
    └── app.js        # Lógica de la app
```

## API Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/models/featured` | Modelos curados |
| POST | `/models/search` | Buscar modelos en HF Hub |
| GET | `/models/{id}/info` | Info de un modelo |
| POST | `/chat/stream` | Chat con streaming SSE |
| POST | `/chat/complete` | Chat sin streaming |
| GET | `/docs` | Swagger UI (FastAPI) |

## Modelos soportados

Cualquier modelo con soporte para **Inference API** de HuggingFace y `chat_completion`. Recomendados:

- `mistralai/Mistral-7B-Instruct-v0.3`
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `google/gemma-2-9b-it`
- `Qwen/Qwen2.5-7B-Instruct`
- `microsoft/Phi-3.5-mini-instruct`
