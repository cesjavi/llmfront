#!/usr/bin/env bash
# LLMFront – Script de inicio
# Uso: ./run.sh [--port 8000] [--token hf_xxx]

PORT=8000
TOKEN=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --port) PORT="$2"; shift 2;;
    --token) TOKEN="$2"; shift 2;;
    *) echo "Opción desconocida: $1"; shift;;
  esac
done

# Crear .env si no existe
if [ ! -f .env ]; then
  cp .env.example .env
  echo "📝 Creado .env desde .env.example"
  if [ -n "$TOKEN" ]; then
    sed -i "s/hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx/$TOKEN/" .env
  fi
fi

# Instalar dependencias si hace falta
if ! python3 -c "import fastapi, huggingface_hub, sse_starlette" 2>/dev/null; then
  echo "📦 Instalando dependencias..."
  pip3 install -r requirements.txt -q
fi

echo ""
echo "╔══════════════════════════════════════╗"
echo "║         LLMFront  🤗                 ║"
echo "╠══════════════════════════════════════╣"
echo "║  URL: http://localhost:$PORT/app       ║"
echo "║  API: http://localhost:$PORT/docs      ║"
echo "╚══════════════════════════════════════╝"
echo ""

PORT=$PORT python3 main.py
