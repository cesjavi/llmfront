// LLMFront - Frontend logic v2.1
let state = {
    view: 'chat',
    modelsTab: 'cloud',
    featuredModels: [],
    activeModel: null, // {id, name, isLocal, supportsVision}
    isLocal: false,
    messages: [],
    pendingImages: [], // [{base64, mimeType}]
    hfToken: localStorage.getItem('hf_token') || '',
    groqToken: localStorage.getItem('groq_token') || '',
    openrouterToken: localStorage.getItem('openrouter_token') || '',
    provider: 'hf',
    systemInfo: null,
    downloading: {},
    quant: 'none',
    cameraStream: null
};

// ─── INIT ───
document.addEventListener('DOMContentLoaded', () => {
    if (state.hfToken) document.getElementById('hfTokenInput').value = state.hfToken;
    if (state.groqToken) document.getElementById('groqTokenInput').value = state.groqToken;
    if (state.openrouterToken) document.getElementById('openrouterTokenInput').value = state.openrouterToken;
    
    document.getElementById('saveTokenBtn').addEventListener('click', saveTokens);
    document.getElementById('changeModelBtn').addEventListener('click', () => switchView('models'));
    document.getElementById('newChatBtn').addEventListener('click', clearChat);
    loadFeaturedModels();
    updateHardwareInfo();
    setInterval(updateHardwareInfo, 10000);
    setInterval(checkDownloadsProgress, 2000);

    // Enter to send
    document.getElementById('userInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    document.getElementById('userInput').addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        document.getElementById('charCount').innerText = `${this.value.length} / 8000`;
        updateSendButton();
    });

    // File input change
    document.getElementById('imageInput').addEventListener('change', handleImageSelection);

    // Paste images from clipboard (Ctrl+V)
    document.addEventListener('paste', handlePaste);

    // Live update from canvas editor
    document.getElementById('canvasCodeEditor').addEventListener('input', function() {
        const iframe = document.getElementById('canvasIframe');
        iframe.srcdoc = this.value;
    });
});

// ─── CORE CHAT ───
async function sendMessage() {
    const input = document.getElementById('userInput');
    const text = input.value.trim();
    if (!text && state.pendingImages.length === 0) return;
    if (!state.activeModel) {
        showToast("Seleccioná un modelo primero", "error");
        switchView('models');
        return;
    }

    const userMsg = { role: 'user', content: text, images: [...state.pendingImages] };
    state.messages.push(userMsg);
    renderMessages();
    
    input.value = '';
    input.style.height = 'auto';
    const imagesToSend = [...state.pendingImages];
    clearPendingImages();
    updateSendButton();

    // Placeholder for AI
    const aiMsg = { role: 'assistant', content: '', thinking: true };
    state.messages.push(aiMsg);
    renderMessages();

    const scroll = () => {
        const wrap = document.getElementById('messagesWrap');
        wrap.scrollTop = wrap.scrollHeight;
    };
    scroll();

    let currentKey = state.hfToken;
    if (state.provider === 'groq') currentKey = state.groqToken;
    if (state.provider === 'openrouter') currentKey = state.openrouterToken;

    const payload = {
        model: state.activeModel.id,
        messages: state.messages.slice(0, -1).map(m => ({ role: m.role, content: m.content })),
        system_prompt: document.getElementById('systemPrompt').value,
        temperature: parseFloat(document.getElementById('temperature').value),
        max_new_tokens: parseInt(document.getElementById('maxNewTokens').value),
        top_p: parseFloat(document.getElementById('topP').value),
        repetition_penalty: parseFloat(document.getElementById('repPenalty').value),
        stream: document.getElementById('streamToggle').checked,
        hf_token: state.hfToken, // For backwards compatibility or direct HF logic
        api_key: currentKey,
        provider: state.provider,
        use_local: state.isLocal,
        images: imagesToSend.length > 0 ? imagesToSend : null
    };

    try {
        const response = await fetch('/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Error en el servidor");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let aiFullText = "";
        let sseBuffer = "";
        aiMsg.thinking = false;

        while (true) {
            const { done, value } = await reader.read();
            if (done) {
                sseBuffer += decoder.decode();
                break;
            }
            
            sseBuffer += decoder.decode(value, { stream: true });
            const events = sseBuffer.split('\n\n');
            sseBuffer = events.pop() || "";

            for (const event of events) {
                for (const line of event.split('\n')) {
                    if (!line.startsWith('data: ')) continue;
                    let data;
                    try {
                        data = JSON.parse(line.substring(6));
                    } catch (e) { console.error("JSON parse error", e); }
                    if (!data) continue;
                    if (data.error) throw new Error(data.error);
                    if (data.token) {
                        aiFullText += data.token;
                        aiMsg.content = aiFullText;
                        renderMessages();
                        scroll();
                    }
                    if (data.done) {
                        aiMsg.thinking = false;
                        if (data.tokens) {
                            aiMsg.tokens = data.tokens;
                        }
                        renderMessages();
                        scroll();
                    }
                }
            }
        }

        if (sseBuffer.trim()) {
            for (const line of sseBuffer.split('\n')) {
                if (!line.startsWith('data: ')) continue;
                let data;
                try {
                    data = JSON.parse(line.substring(6));
                } catch (e) { console.error("JSON parse error", e); }
                if (!data) continue;
                if (data.error) throw new Error(data.error);
                if (data.token) {
                    aiFullText += data.token;
                    aiMsg.content = aiFullText;
                }
                if (data.done) {
                    aiMsg.thinking = false;
                    if (data.tokens) {
                        aiMsg.tokens = data.tokens;
                    }
                }
            }
            renderMessages();
            scroll();
        }
    } catch (err) {
        aiMsg.content = `❌ Error: ${err.message}`;
        aiMsg.thinking = false;
        renderMessages();
        showToast(err.message, "error");
    }
}

function renderMessages() {
    const container = document.getElementById('messages');
    container.innerHTML = '';
    
    if (state.messages.length === 0) {
        container.innerHTML = getWelcomeHtml();
        return;
    }

    state.messages.forEach((msg, idx) => {
        const div = document.createElement('div');
        const isUser = msg.role === 'user';
        div.className = `message ${isUser ? 'user' : 'assistant'}`;
        
        let contentHtml = '';
        if (msg.images && msg.images.length > 0) {
            contentHtml += `<div class="msg-images">`;
            msg.images.forEach(img => {
                contentHtml += `<img src="data:${img.mime_type};base64,${img.base64}" class="msg-img" onclick="expandImage(this.src)" />`;
            });
            contentHtml += `</div>`;
        }
        
        const text = msg.content || '';
        const mdText = msg.thinking ? '<span class="thinking-dots"><span>.</span><span>.</span><span>.</span></span>' : formatMarkdown(text);
        
        contentHtml += `<div class="msg-bubble">${mdText}</div>`;
        if (msg.tokens && !msg.thinking) {
            contentHtml += `<div class="msg-tokens" style="font-size: 0.75rem; color: var(--text-muted); margin-top: 4px; text-align: right;">Tokens: ${msg.tokens.prompt} en prompt / ${msg.tokens.completion} generados</div>`;
        }
        let actionsHtml = '';
        if (!msg.thinking && text) {
            actionsHtml = `<div class="msg-actions">
                <button onclick="copyMessage(this, ${idx})" class="icon-btn" title="Copiar mensaje">📋 Copiar</button>
            </div>`;
        }
        
        div.innerHTML = `
            <div class="msg-avatar">${isUser ? 'Tu' : 'AI'}</div>
            <div class="msg-content">
                ${contentHtml}
                ${actionsHtml}
            </div>
        `;
        container.appendChild(div);
    });
}

function copyMessage(btn, idx) {
    const msg = state.messages[idx];
    if (msg && msg.content) {
        navigator.clipboard.writeText(msg.content).then(() => {
            const originalText = btn.innerHTML;
            btn.innerHTML = '✅ Copiado';
            setTimeout(() => btn.innerHTML = originalText, 2000);
        });
    }
}

function copyCode(btn) {
    const codeEl = btn.parentElement.nextElementSibling.querySelector('code');
    if (codeEl) {
        navigator.clipboard.writeText(codeEl.innerText).then(() => {
            const originalText = btn.innerText;
            btn.innerText = 'Copiado!';
            setTimeout(() => btn.innerText = originalText, 2000);
        });
    }
}

function formatMarkdown(text) {
    const escaped = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    return escaped
        .replace(/```(?:([a-zA-Z0-9_+-]+)\n)?([\s\S]*?)```/g, (match, lang, code) => {
            const l = lang || 'texto';
            const cleanLang = l.toLowerCase();
            const isRenderable = ['html', 'svg', 'xml', 'react', 'javascript', 'js', 'css'].includes(cleanLang);
            const renderBtn = isRenderable ? `<button class="copy-code-btn" onclick="openCanvas(this)" style="color:var(--accent-hf); margin-left:8px; font-weight:bold;">✨ Renderizar</button>` : '';
            return `<div class="code-block-wrapper"><div class="code-header"><span class="code-lang">${l}</span><div><button class="copy-code-btn" onclick="copyCode(this)">Copiar</button>${renderBtn}</div></div><pre><code>${code}</code></pre></div>`;
        })
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        .replace(/^\s*[-*]\s+(.+)$/gm, '<span class="md-list-item">$1</span>')
        .replace(/^\s*(\d+)\.\s+(.+)$/gm, '<strong>$1.</strong> $2')
        .replace(/\n/g, '<br/>');
}

// ─── CANVAS FUNCTIONS ───
function toggleCanvas(forceState) {
    const panel = document.getElementById('canvasPanel');
    if (forceState === undefined) {
        panel.classList.toggle('collapsed');
    } else {
        if (forceState) panel.classList.remove('collapsed');
        else panel.classList.add('collapsed');
    }
}

function switchCanvasTab(tab) {
    document.getElementById('canvasTabPreview').classList.toggle('active', tab === 'preview');
    document.getElementById('canvasTabCode').classList.toggle('active', tab === 'code');
    
    const previewWrap = document.getElementById('canvasPreviewWrap');
    const codeWrap = document.getElementById('canvasCodeWrap');
    
    if (tab === 'preview') {
        previewWrap.classList.add('active');
        codeWrap.classList.remove('active');
        codeWrap.style.display = 'none';
        previewWrap.style.display = 'flex';
    } else {
        codeWrap.classList.add('active');
        previewWrap.classList.remove('active');
        previewWrap.style.display = 'flex';
        codeWrap.style.display = 'flex';
    }
}

function copyCanvasCode() {
    const code = document.getElementById('canvasCodeEditor').value;
    navigator.clipboard.writeText(code).then(() => {
        showToast("Código copiado", "success");
    });
}

function openCanvas(btn) {
    const messageDiv = btn.closest('.message');
    if (!messageDiv) return;
    
    // Find all code blocks in this message
    const blocks = Array.from(messageDiv.querySelectorAll('.code-block-wrapper'));
    
    let htmlCode = '';
    let cssCode = '';
    let jsCode = '';
    let svgCode = '';
    
    blocks.forEach(block => {
        const langSpan = block.querySelector('.code-lang');
        if (!langSpan) return;
        const lang = langSpan.innerText.toLowerCase();
        const code = block.querySelector('code').textContent;
        
        if (lang === 'html') htmlCode = code;
        else if (lang === 'css') cssCode = code;
        else if (lang === 'js' || lang === 'javascript') jsCode = code;
        else if (lang === 'svg') svgCode = code;
    });
    
    let finalCode = '';
    
    // Combine blocks if HTML is present
    if (htmlCode) {
        finalCode = htmlCode;
        if (cssCode) {
            if (finalCode.includes('</head>')) {
                finalCode = finalCode.replace('</head>', `<style>\n${cssCode}\n</style>\n</head>`);
            } else {
                finalCode = `<style>\n${cssCode}\n</style>\n` + finalCode;
            }
        }
        if (jsCode) {
            if (finalCode.includes('</body>')) {
                finalCode = finalCode.replace('</body>', `<script>\n${jsCode}\n</script>\n</body>`);
            } else {
                finalCode += `\n<script>\n${jsCode}\n</script>`;
            }
        }
    } else if (svgCode) {
        finalCode = `<!DOCTYPE html><html><body style="display:flex;justify-content:center;align-items:center;height:100vh;margin:0;">${svgCode}</body></html>`;
    } else if (jsCode && !htmlCode && !cssCode) {
        finalCode = `<!DOCTYPE html><html><body><script>\n${jsCode}\n</script></body></html>`;
    } else {
        // Fallback to the block clicked
        const clickedBlock = btn.closest('.code-block-wrapper');
        const lang = clickedBlock.querySelector('.code-lang').innerText.toLowerCase();
        const code = clickedBlock.querySelector('code').textContent;
        
        if (lang === 'css') {
            finalCode = `<!DOCTYPE html><html><head><style>${code}</style></head><body><h1>Preview CSS</h1></body></html>`;
        } else {
            finalCode = code;
        }
    }
    
    const editor = document.getElementById('canvasCodeEditor');
    editor.value = finalCode;
    document.getElementById('canvasTitleText').innerText = `Preview Canvas`;
    
    const iframe = document.getElementById('canvasIframe');
    
    // Forzar actualización recreando el srcdoc
    iframe.srcdoc = '';
    setTimeout(() => {
        iframe.srcdoc = finalCode;
    }, 10);
    
    switchCanvasTab('preview');
    toggleCanvas(true);
}


function getWelcomeHtml() {
    return `
        <div class="welcome-screen" id="welcomeScreen">
          <div class="welcome-icon">ðŸ¤—</div>
          <h1 class="welcome-title">LLMFront</h1>
          <p class="welcome-sub">Chat con cualquier modelo de Hugging Face</p>
          <div class="welcome-steps">
            <div class="step"><span class="step-num">1</span><span>IngresÃ¡ tu <strong>HF Token</strong> en la barra lateral</span></div>
            <div class="step"><span class="step-num">2</span><span>SeleccionÃ¡ un <strong>modelo</strong> en la secciÃ³n Modelos</span></div>
            <div class="step"><span class="step-num">3</span><span>ConfigurÃ¡ el <strong>contexto</strong> y parÃ¡metros a tu gusto</span></div>
            <div class="step"><span class="step-num">4</span><span>Â¡EmpezÃ¡ a <strong>chatear</strong>!</span></div>
          </div>
        </div>
    `;
}

// ─── VISION & CAMERA ───
function triggerImagePicker() {
    document.getElementById('imageInput').click();
}

async function handleImageSelection(e) {
    const files = Array.from(e.target.files);
    for (const file of files) {
        await processImageFile(file);
    }
    e.target.value = ''; // Reset for next selection
}

async function handlePaste(e) {
    const items = Array.from(e.clipboardData?.items || []);
    const imageItems = items.filter(item => item.type.startsWith('image/'));
    if (imageItems.length === 0) return;

    // Solo activar si el modelo soporta visión
    if (!state.activeModel?.supports_vision) {
        showToast('El modelo activo no soporta imágenes', 'error');
        return;
    }

    e.preventDefault();
    for (const item of imageItems) {
        const file = item.getAsFile();
        if (file) await processImageFile(file);
    }
    showToast(`📋 ${imageItems.length} imagen${imageItems.length > 1 ? 'es' : ''} pegada${imageItems.length > 1 ? 's' : ''}`, 'success');
}

function processImageFile(file) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const base64 = e.target.result.split(',')[1];
            state.pendingImages.push({ base64, mime_type: file.type });
            renderPendingImages();
            updateSendButton();
            resolve();
        };
        reader.readAsDataURL(file);
    });
}

function renderPendingImages() {
    const area = document.getElementById('imageAttachment');
    if (state.pendingImages.length === 0) {
        area.style.display = 'none';
        return;
    }
    area.style.display = 'flex';
    area.style.flexWrap = 'wrap';
    area.style.gap = '8px';
    area.innerHTML = '';
    
    state.pendingImages.forEach((img, idx) => {
        const wrap = document.createElement('div');
        wrap.className = 'pending-img-wrap';
        wrap.style.position = 'relative';
        wrap.innerHTML = `
            <img src="data:${img.mime_type};base64,${img.base64}" style="width:60px;height:60px;object-fit:cover;border-radius:8px;border:1px solid var(--border);" />
            <button onclick="removePendingImage(${idx})" class="remove-img-btn" style="position:absolute;top:-5px;right:-5px;background:var(--danger);color:white;border:none;border-radius:50%;width:18px;height:18px;cursor:pointer;font-size:12px;line-height:18px;text-align:center;">×</button>
        `;
        area.appendChild(wrap);
    });
}

function removePendingImage(idx) {
    state.pendingImages.splice(idx, 1);
    renderPendingImages();
    updateSendButton();
}

function clearPendingImages() {
    state.pendingImages = [];
    renderPendingImages();
}

// Camera controls
async function toggleCameraModal() {
    const modal = document.getElementById('cameraModal');
    if (modal.style.display === 'none') {
        try {
            state.cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
            document.getElementById('cameraVideo').srcObject = state.cameraStream;
            modal.style.display = 'flex';
        } catch (err) {
            showToast("No se pudo acceder a la cámara: " + err.message, "error");
        }
    } else {
        if (state.cameraStream) {
            state.cameraStream.getTracks().forEach(track => track.stop());
            state.cameraStream = null;
        }
        modal.style.display = 'none';
    }
}

function takeSnapshot() {
    const video = document.getElementById('cameraVideo');
    const canvas = document.getElementById('cameraCanvas');
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    const base64 = canvas.toDataURL('image/jpeg').split(',')[1];
    state.pendingImages.push({ base64, mime_type: 'image/jpeg' });
    
    renderPendingImages();
    updateSendButton();
    toggleCameraModal();
    showToast("Foto capturada", "success");
}

// ─── NAVIGATION & UI ───
function switchView(view) {
    state.view = view;
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.getElementById(`view${view.charAt(0).toUpperCase() + view.slice(1)}`).classList.add('active');
    
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(`nav${view.charAt(0).toUpperCase() + view.slice(1)}`).classList.add('active');
}

function switchModelsTab(tab) {
    state.modelsTab = tab;
    document.getElementById('tabCloud').classList.toggle('active', tab === 'cloud');
    document.getElementById('tabLocal').classList.toggle('active', tab === 'local');
    document.getElementById('tabContentCloud').style.display = tab === 'cloud' ? 'block' : 'none';
    document.getElementById('tabContentLocal').style.display = tab === 'local' ? 'block' : 'none';

    if (tab === 'local') {
        if (state.featuredModels.length > 0) {
            renderModels(state.featuredModels, 'downloadModelsGrid');
        }
        loadLocalModels();
    }
}

function updateTokenInputVisibility() {
    const p = document.getElementById('keyProviderSelect').value;
    document.getElementById('hfTokenInput').style.display = p === 'hf' ? 'block' : 'none';
    document.getElementById('groqTokenInput').style.display = p === 'groq' ? 'block' : 'none';
    document.getElementById('openrouterTokenInput').style.display = p === 'openrouter' ? 'block' : 'none';
    
    // hint updates
    const hints = {
        'hf': 'Necesario para modo API. <a href="https://huggingface.co/settings/tokens" target="_blank">Obtener token</a>',
        'groq': 'Necesario para usar Groq. <a href="https://console.groq.com/keys" target="_blank">Obtener API Key</a>',
        'openrouter': 'Necesario para usar OpenRouter. <a href="https://openrouter.ai/keys" target="_blank">Obtener API Key</a>'
    };
    document.getElementById('tokenHint').innerHTML = hints[p];
}

function setProvider(provider) {
    state.provider = provider;
    if (provider === 'local') {
        state.isLocal = true;
        document.getElementById('modeLocalBtn').classList.add('active');
        document.getElementById('footerMode').innerText = 'Local 💻';
    } else {
        state.isLocal = false;
        document.getElementById('modeLocalBtn').classList.remove('active');
        document.getElementById('footerMode').innerText = `API ☁️ (${provider})`;
    }
    updateVisionSupport();
}

function setMode(mode) {
    if (mode === 'local') {
        setProvider('local');
    } else {
        setProvider(document.getElementById('chatProviderSelect').value);
    }
}

function updateModeUI() {
    if (state.isLocal) {
        document.getElementById('modeLocalBtn').classList.add('active');
    } else {
        document.getElementById('modeLocalBtn').classList.remove('active');
    }
}

function saveTokens() {
    state.hfToken = document.getElementById('hfTokenInput').value.trim();
    state.groqToken = document.getElementById('groqTokenInput').value.trim();
    state.openrouterToken = document.getElementById('openrouterTokenInput').value.trim();
    
    localStorage.setItem('hf_token', state.hfToken);
    localStorage.setItem('groq_token', state.groqToken);
    localStorage.setItem('openrouter_token', state.openrouterToken);
    showToast("API Keys guardadas", "success");
}

function updateSlider(inputId, valueId, value, formatFn) {
    document.getElementById(valueId).innerText = formatFn(value);
    if (inputId === 'maxNewTokens') document.getElementById('footerTokens').innerText = value;
    if (inputId === 'temperature') document.getElementById('footerTemp').innerText = value;
}

function applyPreset(preset) {
    const presets = {
        precise: { temperature: 0.2, topP: 0.8, repPenalty: 1.15 },
        balanced: { temperature: 0.7, topP: 0.9, repPenalty: 1.1 },
        creative: { temperature: 1.1, topP: 0.95, repPenalty: 1.05 },
        code: { temperature: 0.3, topP: 0.85, repPenalty: 1.12 }
    };
    const cfg = presets[preset] || presets.balanced;
    setConfigValue('temperature', 'tempValue', cfg.temperature);
    setConfigValue('topP', 'topPValue', cfg.topP);
    setConfigValue('repPenalty', 'repPenValue', cfg.repPenalty);
    showToast("Preset aplicado", "success");
}

function resetConfig() {
    setConfigValue('temperature', 'tempValue', 0.7);
    setConfigValue('maxNewTokens', 'maxTokensValue', 512);
    setConfigValue('topP', 'topPValue', 0.9);
    setConfigValue('repPenalty', 'repPenValue', 1.1);
    document.getElementById('streamToggle').checked = true;
    document.getElementById('systemPrompt').value = "Eres un asistente útil, amable y preciso. Respondes siempre en el idioma del usuario.";
    showToast("Configuración restaurada", "success");
}

function setConfigValue(inputId, valueId, value) {
    document.getElementById(inputId).value = value;
    updateSlider(inputId, valueId, String(value), v => v);
}

function updateSendButton() {
    const btn = document.getElementById('sendBtn');
    const hasText = document.getElementById('userInput').value.trim().length > 0;
    const hasImages = state.pendingImages.length > 0;
    btn.disabled = !(hasText || hasImages);
}

function updateVisionSupport() {
    const btnAttach = document.getElementById('attachImageBtn');
    const btnCamera = document.getElementById('cameraBtn');
    const hint = document.getElementById('visionHint');
    
    const supports = state.activeModel && state.activeModel.supports_vision;
    
    if (supports) {
        btnAttach.style.display = 'flex';
        btnCamera.style.display = 'flex';
        hint.style.display = 'block';
    } else {
        btnAttach.style.display = 'none';
        btnCamera.style.display = 'none';
        hint.style.display = 'none';
    }
}

// ─── MODELS ───
async function loadFeaturedModels() {
    try {
        const res = await fetch('/models/featured');
        const data = await res.json();
        state.featuredModels = data.models || [];
        renderModels(state.featuredModels, 'featuredModelsGrid');
        renderModels(state.featuredModels, 'downloadModelsGrid');
        if (state.modelsTab === 'local') loadLocalModels();
    } catch (e) { console.error(e); }
}

async function loadLocalModels() {
    try {
        const res = await fetch('/models/local');
        const data = await res.json();
        renderLocalModelsList(data);
    } catch (e) { console.error(e); }
}

function renderLocalModelsList(data) {
    const models = data.models || [];
    const list = document.getElementById('localModelsList');
    const count = document.getElementById('hwModels');
    if (count) count.innerText = `📦 Descargados: ${models.length}`;

    if (models.length === 0) {
        list.innerHTML = '<div class="local-empty">No hay modelos descargados aún.<br>Seleccioná un modelo de la lista de destacados y hacé clic en <strong>📥 Descargar local</strong>.</div>';
        return;
    }

    list.innerHTML = '';
    models.forEach(m => {
        const card = document.createElement('div');
        card.className = `local-model-card ${m.is_loaded ? 'is-loaded' : ''} ${m.is_partial ? 'is-partial' : ''}`;

        const isComplete = m.is_complete;
        const actionLabel = m.is_loaded ? 'Usar' : isComplete ? 'Cargar' : '▶ Reanudar';
        const actionClass = m.is_loaded ? 'use' : isComplete ? 'load' : 'resume';
        const statusLabel = m.is_loaded ? 'Cargado' : isComplete ? 'Listo' : 'Incompleto';
        const badgeClass  = m.is_loaded ? 'loaded' : isComplete ? 'downloaded' : 'partial';

        card.innerHTML = `
            <div class="local-model-icon">💻</div>
            <div class="local-model-info">
                <div class="local-model-name">${m.name}</div>
                <div class="local-model-meta">
                    ${m.id} · ${m.size_gb || 0} GB
                    <span class="local-badge ${badgeClass}">${statusLabel}</span>
                </div>
            </div>
            <div class="local-model-actions">
                <button class="local-action-btn ${actionClass}">${actionLabel}</button>
                <button class="local-action-btn delete" title="Eliminar del disco">🗑️</button>
            </div>
        `;

        // Acción principal: cargar/usar si completo, reanudar si parcial
        card.querySelectorAll('.local-action-btn')[0].onclick = () => {
            currentModalModel = m;
            openDownloadModal(m);
            if (isComplete) {
                showLoadStep(m);
            } else {
                // Parcial: mostrar sección de descarga para reanudar
                document.getElementById('dlModalTitle').innerText = 'Reanudar descarga';
                document.getElementById('dlConfirmBtn').innerText = '▶ Reanudar descarga';
            }
        };

        // Botón eliminar
        card.querySelectorAll('.local-action-btn')[1].onclick = () => deleteLocalModel(m);

        list.appendChild(card);
    });
}

function renderModels(models, gridId) {
    const grid = document.getElementById(gridId);
    grid.innerHTML = '';
    models.forEach(m => {
        const card = document.createElement('div');
        const tags = Array.isArray(m.tags) ? m.tags : [];
        card.className = `model-item ${state.activeModel?.id === m.id ? 'selected' : ''}`;
        card.onclick = () => showModelModal(m);
        card.innerHTML = `
            <div class="model-item-header">
                <span class="m-icon">${m.supports_vision ? '👁️' : '📝'}</span>
                <div class="m-info">
                   <div class="m-name">${m.name}</div>
                   <div class="m-id">${m.id}</div>
                </div>
            </div>
            <div class="m-tags">
                ${m.tags.map(t => `<span class="m-tag">${t}</span>`).join('')}
                ${m.is_downloaded ? '<span class="m-tag local-tag">💻 Local</span>' : ''}
            </div>
            <div class="m-desc">${m.description || 'Sin descripción'}</div>
            <div class="m-footer">
                <span>⭐ ${m.likes}</span>
                <span>${m.size_gb ? m.size_gb + ' GB' : ''}</span>
            </div>
        `;
        grid.appendChild(card);
    });
}

function showModelModal(m) {
    currentModalModel = m;
    const modal = document.getElementById('modelConfirmModal');
    document.getElementById('modalModelName').innerText = m.name;
    document.getElementById('modalModelId').innerText = m.id;
    document.getElementById('modalModelDesc').innerText = m.description || 'Sin descripción disponible.';
    document.getElementById('modalModelCapability').innerText = m.capability_label;
    
    const tags = document.getElementById('modalModelTags');
    tags.innerHTML = m.tags.map(t => `<span class="m-tag">${t}</span>`).join('');
    
    document.getElementById('modalConfirmBtn').onclick = () => {
        setActiveModel(m, false);
        closeModal();
    };

    const sizeSpan = document.getElementById('modalModelSize') || document.createElement('div');
    sizeSpan.id = 'modalModelSize';
    sizeSpan.style.marginTop = '10px';
    sizeSpan.style.color = 'var(--text-muted)';
    sizeSpan.style.fontSize = '0.9rem';
    sizeSpan.innerText = m.size_gb ? `Tamaño: ${m.size_gb} GB` : 'Calculando tamaño exacto...';
    document.getElementById('modalModelDesc').after(sizeSpan);
    
    if (!m.size_gb) {
        fetch(`/models/size/${encodeURIComponent(m.id)}`).then(r => r.json()).then(d => {
            if (d.size_gb) {
                m.size_gb = d.size_gb;
                sizeSpan.innerText = `Tamaño exacto: ${d.size_gb} GB`;
            } else {
                sizeSpan.innerText = `Tamaño exacto desconocido`;
            }
        }).catch(() => sizeSpan.innerText = '');
    }

    const dlBtn = document.getElementById('modalDownloadBtn');
    if (m.is_downloaded) {
        dlBtn.innerText = "🧠 Cargar local";
        dlBtn.onclick = () => {
            closeModal();
            openDownloadModal(m);
            showLoadStep(m);
        };
    } else {
        dlBtn.innerText = "📥 Descargar local";
        dlBtn.onclick = () => {
             closeModal();
             openDownloadModal(m);
        };
    }

    modal.style.display = 'flex';
}

function setActiveModel(m, isLocal) {
    state.activeModel = { ...m, isLocal };
    state.isLocal = isLocal;
    
    document.getElementById('activeModelName').innerText = m.name;
    document.getElementById('activeModelId').innerText = m.id;
    document.getElementById('chatModelLabel').innerText = m.name;
    document.getElementById('footerModel').innerText = m.name;
    
    setMode(isLocal ? 'local' : 'api');
    updateVisionSupport();
    showToast(`Modelo: ${m.name}`, "success");
}

function closeModal() {
    document.querySelectorAll('.modal-overlay').forEach(m => m.style.display = 'none');
}

function confirmModelSelect() {
    if (!currentModalModel) return;
    setActiveModel(currentModalModel, false);
    closeModal();
    switchView('chat');
}

function startDownloadFromModal() {
    if (!currentModalModel) return;
    closeModal();
    openDownloadModal(currentModalModel);
    if (currentModalModel.is_downloaded) showLoadStep(currentModalModel);
}

// ─── TOAST & UTILS ───
function showToast(msg, type = 'info') {
    const t = document.getElementById('toast');
    t.innerText = msg;
    t.className = `toast show ${type}`;
    setTimeout(() => t.classList.remove('show'), 3000);
}

async function updateHardwareInfo() {
    try {
        const res = await fetch('/system/info');
        const info = await res.json();
        state.systemInfo = info;
        
        const ram = document.getElementById('hwRam');
        if (ram) ram.innerText = `🖥️ RAM: ${info.ram_available_gb}/${info.ram_total_gb} GB`;
        
        const cuda = document.getElementById('hwCuda');
        if (cuda) {
            cuda.innerText = info.cuda_available ? `⚡ CUDA: ${info.cuda_device} (${info.vram_gb} GB)` : '⚡ CUDA: No detectado';
            cuda.style.color = info.cuda_available ? '#4ade80' : '#94a3b8';
        }
    } catch (e) {}
}

const debounceSearch = (() => {
    let timer;
    return () => {
        clearTimeout(timer);
        timer = setTimeout(searchModels, 800);
    };
})();

async function searchModels() {
    const query = document.getElementById('modelSearchInput').value;
    if (!query) {
        document.getElementById('searchResultsSection').style.display = 'none';
        return;
    }
    
    const grid = document.getElementById('searchModelsGrid');
    grid.innerHTML = '<div class="loading-spinner"><div class="spinner"></div><span>Buscando...</span></div>';
    document.getElementById('searchResultsSection').style.display = 'block';

    try {
        const res = await fetch('/models/search', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                query,
                limit: 10,
                task: document.getElementById('taskFilter').value,
                size_filter: document.getElementById('sizeFilter').value,
                use_ai_search: document.getElementById('aiSearchToggle').checked
            })
        });
        const data = await res.json();
        renderModels(data.models, 'searchModelsGrid');
    } catch (e) {
        grid.innerHTML = `<div class="error-msg">Error: ${e.message}</div>`;
    }
}

// ─── DOWNLOAD & LOCAL LOAD ───
let currentModalModel = null;
function openDownloadModal(m) {
    currentModalModel = m;
    const modal = document.getElementById('downloadModal');
    document.getElementById('dlModalModelId').innerText = m.id;
    document.getElementById('dlModalTitle').innerText = m.is_downloaded ? "Cargar modelo en memoria" : "Descargar modelo";
    
    // Reset view
    document.getElementById('dlSectionDownload').style.display = m.is_downloaded ? 'none' : 'block';
    document.getElementById('dlSectionLoad').style.display = 'none';
    document.getElementById('dlQuantSection').style.display = 'block';
    document.getElementById('dlModalActions').style.display = 'flex';
    document.getElementById('dlModalActionsLoad').style.display = 'none';
    
    modal.style.display = 'flex';
}

function showLoadStep(m) {
    document.getElementById('dlModalTitle').innerText = "Cargar modelo en memoria";
    document.getElementById('dlSectionDownload').style.display = 'none';
    document.getElementById('dlSectionLoad').style.display = 'block';
    document.getElementById('dlModalActions').style.display = 'none';
    document.getElementById('dlModalActionsLoad').style.display = 'flex';
    document.getElementById('dlLoadMsg').innerText = m.is_loaded ? "Modelo cargado. Podés usarlo en el chat." : "Listo para cargar.";
    document.getElementById('dlLoadPct').innerText = m.is_loaded ? "100%" : "0%";
    document.getElementById('dlLoadBar').style.width = m.is_loaded ? "100%" : "0%";
}

function setQuant(q) {
    state.quant = q;
    document.querySelectorAll('.quant-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(`q${q.charAt(0).toUpperCase() + q.slice(1).toLowerCase()}`).classList.add('active');
}

async function confirmDownload() {
    try {
        const res = await fetch('/models/download', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ model_id: currentModalModel.id })
        });
        const data = await res.json();
        showToast("Descarga iniciada", "info");
        state.downloading[currentModalModel.id] = true;
        document.getElementById('dlCancelBtn').style.display = 'inline-block';
        // Conectar SSE para progress en tiempo real
        listenDownloadProgress(currentModalModel.id);
    } catch (e) { showToast(e.message, "error"); }
}

async function cancelDownload() {
    if (!currentModalModel) return;
    try {
        await fetch(`/models/download/cancel?model_id=${encodeURIComponent(currentModalModel.id)}`, { method: 'POST' });
        showToast("Cancelando descarga...", "info");
        document.getElementById('dlCancelBtn').style.display = 'none';
        delete state.downloading[currentModalModel.id];
    } catch (e) { showToast(e.message, "error"); }
}

function listenDownloadProgress(modelId) {
    const encodedId = modelId.split('/').map(encodeURIComponent).join('/');
    const es = new EventSource(`/models/download/${encodedId}/progress`);

    es.onmessage = (e) => {
        let data;
        try { data = JSON.parse(e.data); } catch { return; }

        if (data._end) { es.close(); return; }

        const dl  = data.download || {};
        const ld  = data.load    || {};

        // Actualizar barra de descarga si el modal está abierto para este modelo
        const modalId = document.getElementById('dlModalModelId')?.innerText;
        const modalOpen = document.getElementById('downloadModal')?.style.display !== 'none';

        if (modalOpen && modalId === modelId) {
            // Progreso de descarga
            if (dl.size_downloaded_gb !== undefined) {
                const pct = dl.progress || 0;
                document.getElementById('dlDownloadBar').style.width = pct + '%';
                document.getElementById('dlDownloadPct').innerText = pct + '%';
                document.getElementById('dlDownloadMsg').innerText = dl.message || '';
                const gb = (dl.size_downloaded_gb || 0).toFixed(2);
                document.getElementById('dlDownloadSize').innerText = `${gb} GB descargados`;
            }
            // Progreso de carga en memoria
            if (ld.status === 'loading' || ld.status === 'loaded') {
                document.getElementById('dlSectionLoad').style.display = 'block';
                const lpct = ld.progress || 0;
                document.getElementById('dlLoadBar').style.width = lpct + '%';
                document.getElementById('dlLoadPct').innerText = lpct + '%';
                document.getElementById('dlLoadMsg').innerText = ld.message || '';
            }
        }

        // Descarga completada
        if (dl.status === 'done') {
            delete state.downloading[modelId];
            // Refrescar listas
            loadLocalModels();
            loadFeaturedModels();
            // Si el modal sigue abierto para este modelo, ir al paso de carga
            if (modalOpen && modalId === modelId) {
                currentModalModel = { ...currentModalModel, is_downloaded: true, is_complete: true };
                showLoadStep(currentModalModel);
                showToast('✅ Descarga completa: ' + modelId.split('/').pop(), 'success');
            } else {
                showToast('✅ Descarga completa: ' + modelId.split('/').pop(), 'success');
            }
        }

        // Carga en memoria completada
        if (ld.status === 'loaded' && modalOpen && modalId === modelId) {
            document.getElementById('dlLoadMsg').innerText = '✓ Listo para usar';
            document.getElementById('dlLoadBar').style.width = '100%';
            document.getElementById('dlLoadPct').innerText = '100%';
            document.getElementById('dlModalActionsLoad').style.display = 'flex';
        }

        if (dl.status === 'error') {
            delete state.downloading[modelId];
            showToast('❌ Error en descarga: ' + (dl.message || ''), 'error');
            es.close();
        }
    };

    es.onerror = () => es.close();
}

async function loadLocalModel() {
    const btn = document.getElementById('dlLoadBtn');
    btn.disabled = true;
    btn.innerText = "⏳ Cargando...";

    document.getElementById('dlSectionLoad').style.display = 'block';
    document.getElementById('dlLoadMsg').innerText = 'Iniciando carga en memoria...';
    document.getElementById('dlLoadBar').style.width = '0%';
    document.getElementById('dlLoadPct').innerText = '0%';

    try {
        const res = await fetch('/models/load', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                model_id: currentModalModel.id,
                quantization: state.quant,
                device: 'auto'
            })
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Error ${res.status}`);
        }
        showToast('🧠 Cargando modelo en memoria...', 'info');
        // Escuchar SSE de progreso (mismo endpoint que descarga, cubre estado de carga)
        listenLoadProgress(currentModalModel.id);
    } catch (e) {
        showToast(e.message, 'error');
        btn.disabled = false;
        btn.innerText = '🧠 Cargar en memoria';
    }
}

function listenLoadProgress(modelId) {
    const encodedId = modelId.split('/').map(encodeURIComponent).join('/');
    const es = new EventSource(`/models/download/${encodedId}/progress`);

    es.onmessage = (e) => {
        let data;
        try { data = JSON.parse(e.data); } catch { return; }
        if (data._end) { es.close(); return; }

        const ld = data.load || {};
        const modalId   = document.getElementById('dlModalModelId')?.innerText;
        const modalOpen = document.getElementById('downloadModal')?.style.display !== 'none';

        if (modalOpen && modalId === modelId) {
            const lpct = ld.progress || 0;
            document.getElementById('dlLoadBar').style.width = lpct + '%';
            document.getElementById('dlLoadPct').innerText = lpct + '%';
            if (ld.message) document.getElementById('dlLoadMsg').innerText = ld.message;
        }

        if (ld.status === 'loaded') {
            es.close();
            // Refrescar lista local
            loadLocalModels();
            loadFeaturedModels();
            showToast('✅ Modelo cargado en memoria', 'success');
            if (modalOpen && modalId === modelId) {
                document.getElementById('dlLoadMsg').innerText = '✓ Modelo cargado y listo';
                document.getElementById('dlLoadBar').style.width = '100%';
                document.getElementById('dlLoadPct').innerText = '100%';
                document.getElementById('dlModalActionsLoad').style.display = 'flex';
                document.getElementById('dlLoadBtn').disabled = false;
                document.getElementById('dlLoadBtn').innerText = '🧠 Cargar en memoria';
            }
        }

        if (ld.status === 'error') {
            es.close();
            const msg = ld.message || 'Error desconocido';
            showToast('❌ Error al cargar: ' + msg, 'error');
            if (modalOpen && modalId === modelId) {
                document.getElementById('dlLoadMsg').innerText = '❌ ' + msg;
                document.getElementById('dlLoadBtn').disabled = false;
                document.getElementById('dlLoadBtn').innerText = '🧠 Reintentar';
            }
        }
    };

    es.onerror = () => es.close();
}

function useLocalModel() {
    setActiveModel(currentModalModel, true);
    closeDownloadModal();
    switchView('chat');
}

function closeDownloadModal() {
    document.getElementById('downloadModal').style.display = 'none';
}

async function deleteLocalModel(m) {
    const label = m.name || m.id;
    if (!confirm(`¿Eliminar "${label}" del disco? Esta acción no se puede deshacer.`)) return;

    try {
        const res = await fetch(`/models/local/${encodeURIComponent(m.id)}`, { method: 'DELETE' });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Error ${res.status}`);
        }
        showToast(`🗑️ "${label}" eliminado`, 'success');
        // Si era el modelo activo, limpiar selección
        if (state.activeModel?.id === m.id) {
            state.activeModel = null;
            state.isLocal = false;
            document.getElementById('activeModelName').innerText = 'Sin seleccionar';
            document.getElementById('activeModelId').innerText = '—';
            document.getElementById('chatModelLabel').innerText = 'Seleccioná un modelo para comenzar';
            updateVisionSupport();
        }
        loadLocalModels();
        loadFeaturedModels(); // Refrescar badge "Local" en la grilla
    } catch (e) {
        showToast(e.message, 'error');
    }
}

function checkDownloadsProgress() {
    // Refrescar lista local si hay descargas activas en segundo plano
    if (Object.keys(state.downloading).length > 0) {
        loadLocalModels();
    }
}

/**** SIDEBAR TOGGLE ****/
function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('collapsed');
}

/**** EXPORT/CLEAR ****/
function clearChat() {
    if (confirm("¿Limpiar historial de chat?")) {
        state.messages = [];
        renderMessages();
    }
}

function exportChat() {
    const data = JSON.stringify(state.messages, null, 2);
    const blob = new Blob([data], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat-export-${new Date().getTime()}.json`;
    a.click();
}
