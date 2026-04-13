const I18N_DICT = {
"Modelos en la nube (API)": "Cloud Models (API)",
"Modelos locales": "Local Models",
"Buscar modelo (ej: mistral, llama, qwen...)": "Search model (e.g., mistral, llama, qwen...)",
"Text Generation": "Text Generation",
"Conversational": "Conversational",
"Text-to-Text": "Text-to-Text",
"Summarization": "Summarization",
"Translation": "Translation",
"Buscar": "Search",
"MODELOS DESTACADOS": "FEATURED MODELS",
"RESULTADOS DE BÚSQUEDA": "SEARCH RESULTS",
"MODELOS DESCARGADOS": "DOWNLOADED MODELS",
"No hay modelos descargados aún.": "No downloaded models yet.",
"Seleccioná un modelo de la lista de destacados y hacé clic en ": "Select a model from the featured list and click on ",
"Descargar local": "Download Local",
"DESCARGAR MODELO": "DOWNLOAD MODEL",
"Configuración del Contexto": "Context Configuration",
"Ajustá el comportamiento del modelo y los parámetros de generación.": "Adjust the model's behavior and generation parameters.",
"Contexto base": "Base Context",
"Ej: Sos un asistente experto en programación Python. Respondé siempre en español, de forma clara y con ejemplos de código cuando sea necesario.": "Ex: You are an expert Python programming assistant. Always answer in English, clearly and with code examples when necessary.",
"Eres un asistente útil, amable y preciso. Respondes siempre en el idioma del usuario.": "You are a helpful, polite, and precise assistant. You always answer in the user's language.",
"Define la personalidad y rol del modelo. Se envía como mensaje de sistema antes de la conversación.": "Defines the personality and role of the model. Sent as a system message before the conversation.",
"Idioma de respuesta (Atajos)": "Response Language (Shortcuts)",
"Español": "Spanish",
"English": "English",
"Temperatura": "Temperature",
"0.01 (Determinístico)": "0.01 (Deterministic)",
"2.0 (Creativo)": "2.0 (Creative)",
"Valores bajos: respuestas más precisas y repetibles. Valores altos: más creatividad y variedad.": "Low values: more precise and repeatable answers. High values: more creativity and variety.",
"Tokens máximos": "Max Tokens",
"Longitud máxima de la respuesta del modelo (en tokens, ~¾ palabras c/u).": "Maximum length of the model's response (in tokens, ~¾ words each).",
"Top P (nucleus sampling)": "Top P (nucleus sampling)",
"Controla la diversidad del vocabulario. 0.9 considera el top 90% de tokens por probabilidad.": "Controls vocabulary diversity. 0.9 considers the top 90% of tokens by probability.",
"Penalización de repetición": "Repetition Penalty",
"1.0 (Sin penalizar)": "1.0 (No penalty)",
"2.0 (Máximo)": "2.0 (Maximum)",
"Reduce la tendencia del modelo a repetir las mismas frases.": "Reduces the model's tendency to repeat the same phrases.",
"Streaming": "Streaming",
"Muestra las respuestas en tiempo real token por token. Desactivar para recibir la respuesta completa de una vez.": "Shows responses in real-time token by token. Disable to receive the full response at once.",
"Presets rápidos": "Quick Presets",
"Preciso": "Precise",
"Balanceado": "Balanced",
"Creativo": "Creative",
"Restaurar valores por defecto": "Restore Default Values",
"Chat": "Chat",
"Modelos": "Models",
"Configuración": "Configuration",
"Escribe tu mensaje...": "Type your message...",
"chatear": "chat",
"selecciona": "select",
"descarga": "download",
"Modelo": "Model",
"Modo": "Mode",
"Max tokens": "Max tokens",
"Temp": "Temp",
"Local": "Local",
"Cargando...": "Loading...",
"Sin modelo local cargado": "No local model loaded",
"Descargar": "Download",
"Cargar en memoria": "Load into memory",
"Iniciar descarga": "Start Download",
"Cerrar": "Close",
"Listo, cerrar": "Done, close",
"Reanudar": "Resume",
"Usar": "Use",
"Descargando": "Downloading",
"Cargando tokenizer...": "Loading tokenizer...",
"Esperando...": "Waiting...",
"Completo": "Full",
"Cancelar": "Cancel",
"Eliminar archivos parciales": "Delete partial files"
};

function getDefaultAppLang() {
    let stored = localStorage.getItem('appLang');
    if (stored) return stored;
    let browserLang = navigator.language || navigator.userLanguage || "es";
    return browserLang.toLowerCase().startsWith('es') ? 'es' : 'en';
}
let currentLang = getDefaultAppLang();

function I18N(text) {
    if (currentLang === 'es') return text;
    // Remove emojis for matching if needed, or exact match
    return I18N_DICT[text] || text;
}

function tWalk(node) {
    if (node.nodeType === 3) { // Text node
        let text = node.nodeValue.trim();
        if (text && I18N_DICT[text]) {
            // Save original es text to a data attribute if not already
            if (!node.parentElement.hasAttribute('data-orig')) {
                node.parentElement.setAttribute('data-orig', text);
            }
            node.nodeValue = node.nodeValue.replace(text, I18N(text));
        } else if (node.parentElement.hasAttribute('data-orig')) {
             let orig = node.parentElement.getAttribute('data-orig');
             if(orig) node.nodeValue = node.nodeValue.replace(text, I18N(orig));
        }
    } else {
        if(node.hasAttribute && node.hasAttribute('placeholder')) {
             let p = node.getAttribute('placeholder');
             if(!node.hasAttribute('data-orig-ph')) node.setAttribute('data-orig-ph', p);
             node.setAttribute('placeholder', I18N(node.getAttribute('data-orig-ph')));
        }
        for (let child of node.childNodes) {
            if(['SCRIPT', 'STYLE'].includes(node.tagName)) continue;
            tWalk(child);
        }
    }
}

function changeAppLanguage(lang) {
    currentLang = lang;
    localStorage.setItem('appLang', lang);
    tWalk(document.body);
    // Refresh dynamic content
    if(window.loadLocalModels) loadLocalModels();
    updateModeUI();
}

document.addEventListener('DOMContentLoaded', () => {
    let sel = document.getElementById('appLangSelect');
    if(sel) sel.value = currentLang;
    if(currentLang !== 'es') tWalk(document.body);
});
