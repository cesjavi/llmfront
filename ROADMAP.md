# LLMFront - Roadmap y Próximas Mejoras

Este documento contiene una lista de ideas evaluadas para potenciar las capacidades y la experiencia de usuario en LLMFront.

## 1. 📚 RAG Simple (Chatear con tus Documentos)
* **Qué es:** Permitir arrastrar archivos **PDF, TXT, CSV o código** directamente al chat.
* **Cómo implementarlo:** Al subir un archivo, el frontend o backend extrae el texto. Se puede usar un modelo ligero de embeddings (ej. `all-MiniLM-L6-v2`) o una base de datos vectorial local (como ChromaDB o extensión SQLite) para que el LLM pueda buscar y responder basándose en el contexto del documento, sin necesidad de pegarlo entero en el prompt.

## 2. 🌐 Búsqueda Web (Estilo Perplexity / SearchGPT)
* **Qué es:** Darle al modelo la capacidad de buscar información actual en internet antes de elaborar su respuesta.
* **Cómo implementarlo:** Integrar la API de **Tavily**, **DuckDuckGo** o crear un scraper simple en Python. En la UI, se agregaría un switch ("🌐 Búsqueda Web"). Si está activo, el backend hace una búsqueda de la pregunta, extrae texto de los primeros resultados y se lo inyecta de forma invisible en el sistema al LLM.

## 3. 💾 Historial de Chats Persistente (Barra lateral)
* **Qué es:** Evitar perder la conversación activa al refrescar el navegador.
* **Cómo implementarlo:** Utilizar `localStorage` o `IndexedDB` en el navegador para almacenar las sesiones localmente de forma segura. Alternativamente, guardar el historial en una base de datos SQLite pequeña con FastAPI. Se requeriría agregar una lista en la barra lateral con las conversaciones previas.

## 4. 🎙️ Interfaz por Voz (Speech-to-Text y Text-to-Speech)
* **Qué es:** Poder hablarle al modelo con el micrófono del dispositivo y que te conteste leyendo en voz alta.
* **Cómo implementarlo:** Hacer uso de la **Web Speech API** que viene integrada de forma nativa (y gratis) en los navegadores modernos. Requerirá botones adicionales en la caja de chat y eventos JS para iniciar/detener el dictado y la lectura.

## 5. 🛠️ Ejecución de Herramientas (Function Calling)
* **Qué es:** Permitir que el LLM local pueda interactuar dinámicamente con tu computadora.
* **Cómo implementarlo:** Aprovechar que el backend corre en Python y definir funciones de utilidad como `ver_la_hora()`, `leer_archivo_local()`, etc. Luego, usar modelos entrenados para herramientas (ej. Llama 3 o Qwen). El modelo pide ejecutar la función, el backend de Python la ejecuta y le devuelve el resultado para que termine su respuesta.
