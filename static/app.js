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
    togetherToken: localStorage.getItem('together_token') || '',
    deepinfraToken: localStorage.getItem('deepinfra_token') || '',
    fireworksToken: localStorage.getItem('fireworks_token') || '',
    basetenToken: localStorage.getItem('baseten_token') || '',
    nvidiaToken: localStorage.getItem('nvidia_token') || '',
    provider: localStorage.getItem('selected_provider') || 'hf',
    searchProvider: localStorage.getItem('selected_provider') || 'hf',
    apiChecks: {},
    systemInfo: null,
    downloading: {},
    quant: 'none',
    cameraStream: null
};

async function preloadConfigBackedProviders() {
    try {
        const res = await fetch('/config');
        const cfg = await res.json();
        const configMap = [
            ['hfToken', 'hfTokenInput', cfg.hf_available],
            ['groqToken', 'groqTokenInput', cfg.groq_available],
            ['openrouterToken', 'openrouterTokenInput', cfg.openrouter_available],
            ['togetherToken', 'togetherTokenInput', cfg.together_available],
            ['deepinfraToken', 'deepinfraTokenInput', cfg.deepinfra_available],
            ['fireworksToken', 'fireworksTokenInput', cfg.fireworks_available],
            ['basetenToken', 'basetenTokenInput', cfg.baseten_available],
            ['nvidiaToken', 'nvidiaTokenInput', cfg.nvidia_available],
        ];

        configMap.forEach(([stateKey, inputId, available]) => {
            const input = document.getElementById(inputId);
            if (!input) return;
            if (!state[stateKey] && available) {
                input.placeholder = 'Configurada en .env';
                input.dataset.fromEnv = 'true';
            }
        });
    } catch (e) {
        console.warn('No se pudo leer /config para precargar providers', e);
    }
}

// ─── INIT ───
document.addEventListener('DOMContentLoaded', () => {
    if (state.hfToken) document.getElementById('hfTokenInput').value = state.hfToken;
    if (state.groqToken) document.getElementById('groqTokenInput').value = state.groqToken;
    if (state.openrouterToken) document.getElementById('openrouterTokenInput').value = state.openrouterToken;
    if (state.togetherToken) document.getElementById('togetherTokenInput').value = state.togetherToken;
    if (state.deepinfraToken) document.getElementById('deepinfraTokenInput').value = state.deepinfraToken;
    if (state.fireworksToken) document.getElementById('fireworksTokenInput').value = state.fireworksToken;
    if (state.basetenToken) document.getElementById('basetenTokenInput').value = state.basetenToken;
    if (state.nvidiaToken) document.getElementById('nvidiaTokenInput').value = state.nvidiaToken;
    updateTokenInputVisibility();
    document.getElementById('searchProviderFilter').value = state.searchProvider;
    document.getElementById('configProviderSelect').value = state.provider;
    document.getElementById('chatProviderSelect').value = state.provider;
    setProvider(state.provider);
    preloadConfigBackedProviders();
    
    document.getElementById('saveTokenBtn').addEventListener('click', saveTokens);
    document.getElementById('changeModelBtn').addEventListener('click', () => switchView('models'));
    document.getElementById('newChatBtn').addEventListener('click', clearChat);
    loadFeaturedModels();
    loadTrendingModels();
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
    document.getElementById('fileInput').addEventListener('change', handleFileSelection);

    // Paste images from clipboard (Ctrl+V)
    document.addEventListener('paste', handlePaste);

    // Live update from canvas editor
    document.getElementById('canvasCodeEditor').addEventListener('input', function() {
        const iframe = document.getElementById('canvasIframe');
        // Usar un pequeño delay o reset para asegurar la actualización en algunos navegadores
        iframe.srcdoc = '';
        const val = this.value;
        setTimeout(() => {
            iframe.srcdoc = val;
        }, 5);
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
    clearPendingFiles();
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
    if (state.provider === 'together') currentKey = state.togetherToken;
    if (state.provider === 'deepinfra') currentKey = state.deepinfraToken;
    if (state.provider === 'fireworks') currentKey = state.fireworksToken;
    if (state.provider === 'baseten') currentKey = state.basetenToken;
    if (state.provider === 'nvidia') currentKey = state.nvidiaToken;

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
                        if (data.is_reasoning) {
                            aiMsg.reasoning = (aiMsg.reasoning || '') + data.token;
                        } else {
                            aiFullText += data.token;
                            aiMsg.content = aiFullText;
                        }
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
                    if (data.is_reasoning) {
                        aiMsg.reasoning = (aiMsg.reasoning || '') + data.token;
                    } else {
                        aiFullText += data.token;
                        aiMsg.content = aiFullText;
                    }
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
        let mdText = msg.thinking ? '<span class="thinking-dots"><span>.</span><span>.</span><span>.</span></span>' : formatMarkdown(text);
        
        if (msg.reasoning) {
            mdText = `<div class="msg-reasoning">
                <div class="reasoning-header">Razonamiento</div>
                <div class="reasoning-body">${formatMarkdown(msg.reasoning)}</div>
            </div>` + mdText;
        }

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
    const wrapper = btn.closest('.code-block-wrapper');
    const codeEl = wrapper ? wrapper.querySelector('code') : null;
    if (codeEl) {
        // Usar textContent para obtener el texto puro sin formato HTML
        const textToCopy = codeEl.textContent;
        navigator.clipboard.writeText(textToCopy).then(() => {
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
          <div class="welcome-icon">LLM</div>
          <h1 class="welcome-title">LLMFront</h1>
          <p class="welcome-sub">Chat con cualquier modelo de Hugging Face</p>
          <div class="welcome-steps">
            <div class="step"><span class="step-num">1</span><span>Ingresa tu <strong>HF Token</strong> en la barra lateral</span></div>
            <div class="step"><span class="step-num">2</span><span>Selecciona un <strong>modelo</strong> en la seccion Modelos</span></div>
            <div class="step"><span class="step-num">3</span><span>Configura el <strong>contexto</strong> y parametros a tu gusto</span></div>
            <div class="step"><span class="step-num">4</span><span>Empieza a <strong>chatear</strong></span></div>
          </div>
        </div>
    `;
}

// ─── VISION & RAG & CAMERA ───
function triggerFilePicker() {
    document.getElementById('fileInput').click();
}

async function handleFileSelection(e) {
    const files = Array.from(e.target.files);
    for (const file of files) {
        if (file.type.startsWith('image/')) {
            if (!state.activeModel?.supports_vision) {
                showToast(`El modelo ${state.activeModel?.id || ''} no soporta imágenes.`, 'error');
                continue;
            }
            await processImageFile(file);
        } else {
            await processRagFile(file);
        }
    }
    e.target.value = ''; // Reset for next selection
}

async function processRagFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    showToast(`Subiendo ${file.name}...`, 'info');
    try {
        const res = await fetch('/rag/upload', {
            method: 'POST',
            body: formData
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Error subiendo archivo');
        }
        const data = await res.json();
        state.pendingDocs = state.pendingDocs || [];
        state.pendingDocs.push({ filename: file.name, doc_id: data.doc_id });
        renderPendingFiles();
        updateSendButton();
        showToast(`Documento indexado: ${file.name}`, 'success');
    } catch(err) {
        showToast(err.message, 'error');
    }
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
            renderPendingFiles();
            updateSendButton();
            resolve();
        };
        reader.readAsDataURL(file);
    });
}

function renderPendingFiles() {
    const area = document.getElementById('imageAttachment');
    const hasImages = state.pendingImages && state.pendingImages.length > 0;
    const hasDocs = state.pendingDocs && state.pendingDocs.length > 0;
    
    if (!hasImages && !hasDocs) {
        area.style.display = 'none';
        return;
    }
    area.style.display = 'flex';
    area.style.flexWrap = 'wrap';
    area.style.gap = '8px';
    area.innerHTML = '';
    
    if (hasImages) {
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
    
    if (hasDocs) {
        state.pendingDocs.forEach((doc, idx) => {
            const wrap = document.createElement('div');
            wrap.className = 'pending-doc-wrap';
            wrap.style.position = 'relative';
            wrap.style.background = 'var(--bg-secondary)';
            wrap.style.padding = '8px 12px';
            wrap.style.borderRadius = '8px';
            wrap.style.border = '1px solid var(--border)';
            wrap.style.display = 'flex';
            wrap.style.alignItems = 'center';
            wrap.style.gap = '6px';
            wrap.innerHTML = `
                <span>📄</span>
                <span style="font-size:12px; max-width:100px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${doc.filename}</span>
                <button onclick="removePendingDoc(${idx})" class="remove-img-btn" style="position:absolute;top:-5px;right:-5px;background:var(--danger);color:white;border:none;border-radius:50%;width:18px;height:18px;cursor:pointer;font-size:12px;line-height:18px;text-align:center;">×</button>
            `;
            area.appendChild(wrap);
        });
    }
}

function removePendingImage(idx) {
    state.pendingImages.splice(idx, 1);
    renderPendingFiles();
    updateSendButton();
}

function removePendingDoc(idx) {
    state.pendingDocs.splice(idx, 1);
    renderPendingFiles();
    updateSendButton();
}

function clearPendingFiles() {
    state.pendingImages = [];
    state.pendingDocs = [];
    renderPendingFiles();
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
    const providers = ['hf', 'groq', 'openrouter', 'together', 'deepinfra', 'fireworks', 'baseten', 'nvidia'];
    
    // Ocultar todos y mostrar solo el seleccionado
    providers.forEach(id => {
        const el = document.getElementById(id + 'TokenInput');
        if (el) el.style.display = (p === id) ? 'block' : 'none';
    });
    
    const hints = {
        'hf': 'Necesario para modo API. <a href="https://huggingface.co/settings/tokens" target="_blank">Obtener token</a>',
        'groq': 'Necesario para usar Groq. <a href="https://console.groq.com/keys" target="_blank">Obtener API Key</a>',
        'openrouter': 'Necesario para usar OpenRouter. <a href="https://openrouter.ai/keys" target="_blank">Obtener API Key</a>',
        'together': 'Necesario para usar Together AI. <a href="https://api.together.xyz/settings/api-keys" target="_blank">Obtener API Key</a>',
        'deepinfra': 'Necesario para usar DeepInfra. <a href="https://deepinfra.com/dash/api_keys" target="_blank">Obtener token</a>',
        'fireworks': 'Necesario para usar Fireworks AI. <a href="https://fireworks.ai/account/api-keys" target="_blank">Obtener API Key</a>',
        'baseten': 'Necesario para usar Baseten. <a href="https://app.baseten.co/settings/api_keys" target="_blank">Obtener API Key</a>',
        'nvidia': 'Necesario para usar NVIDIA. <a href="https://build.nvidia.com/" target="_blank">Obtener API Key</a>'
    };
    
    const hintEl = document.getElementById('tokenHint');
    if (hintEl) {
        hintEl.innerHTML = hints[p] || hints['hf'] || '';
    }
}

function setProvider(provider) {
    state.provider = provider;
    localStorage.setItem('selected_provider', provider);
    // Sincronizar todos los selectores
    const selectors = ['chatProviderSelect', 'configProviderSelect', 'searchProviderFilter', 'keyProviderSelect'];
    selectors.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.value = provider;
    });

    if (provider !== 'local') {
        state.searchProvider = provider;
    }
    
    updateTokenInputVisibility();
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

function syncSearchProvider() {
    const provider = document.getElementById('searchProviderFilter').value;
    state.searchProvider = provider;
    setProvider(provider);
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
    state.togetherToken = document.getElementById('togetherTokenInput').value.trim();
    state.deepinfraToken = document.getElementById('deepinfraTokenInput').value.trim();
    state.fireworksToken = document.getElementById('fireworksTokenInput').value.trim();
    state.basetenToken = document.getElementById('basetenTokenInput').value.trim();
    state.nvidiaToken = document.getElementById('nvidiaTokenInput').value.trim();
    state.apiChecks = {};
    
    localStorage.setItem('hf_token', state.hfToken);
    localStorage.setItem('groq_token', state.groqToken);
    localStorage.setItem('openrouter_token', state.openrouterToken);
    localStorage.setItem('together_token', state.togetherToken);
    localStorage.setItem('deepinfra_token', state.deepinfraToken);
    localStorage.setItem('fireworks_token', state.fireworksToken);
    localStorage.setItem('baseten_token', state.basetenToken);
    localStorage.setItem('nvidia_token', state.nvidiaToken);
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
    const btnAttach = document.getElementById('attachBtn');
    const btnCamera = document.getElementById('cameraBtn');
    const hint = document.getElementById('visionHint');
    
    const supports = state.activeModel && state.activeModel.supports_vision;
    
    // attachBtn (clip) stays visible because it also handles RAG documents
    if (btnAttach) btnAttach.style.display = 'flex';

    if (supports) {
        if (btnCamera) btnCamera.style.display = 'flex';
        if (hint) hint.style.display = 'block';
    } else {
        if (btnCamera) btnCamera.style.display = 'none';
        if (hint) hint.style.display = 'none';
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

async function loadTrendingModels() {
    const grid = document.getElementById('trendingModelsGrid');
    grid.innerHTML = '<div class="loading-spinner"><div class="spinner"></div><span>Cargando tendencias...</span></div>';
    try {
        const res = await fetch('/models/trending');
        const data = await res.json();
        if (data.models) {
            renderModels(data.models, 'trendingModelsGrid');
        } else {
            grid.innerHTML = '<div class="local-empty">No se pudieron cargar las tendencias.</div>';
        }
    } catch (e) { 
        console.error(e);
        grid.innerHTML = '<div class="local-empty">Error cargando tendencias.</div>';
    }
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
    const provider = state.searchProvider || state.provider || 'hf';
    const showApiStatus = gridId === 'searchModelsGrid';
    models.forEach(m => {
        const card = document.createElement('div');
        const tags = Array.isArray(m.tags) ? m.tags : [];
        const cacheKey = `${provider}:${m.id}`;
        const apiInfo = state.apiChecks[cacheKey] || null;
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
            ${showApiStatus ? `
                <div class="m-api-status ${apiInfo ? apiInfo.status : 'pending'}">
                    <span class="m-api-status-label">${provider.toUpperCase()}</span>
                    <span class="m-api-status-value">${apiInfo ? apiInfo.label : 'Verificando...'}</span>
                </div>
            ` : ''}
            <div class="m-footer">
                <span>⭐ ${m.likes}</span>
                <span>${m.size_gb ? m.size_gb + ' GB' : ''}</span>
            </div>
            <div class="m-actions">
                <button class="model-details-btn" type="button">Ver detalles</button>
            </div>
        `;
        const detailsBtn = card.querySelector('.model-details-btn');
        if (detailsBtn) {
            detailsBtn.onclick = (event) => {
                event.stopPropagation();
                showModelModal(m);
            };
        }
        grid.appendChild(card);
    });

    if (showApiStatus) {
        enrichSearchResultApiStatus(models, provider);
    }
}

function getProviderApiKey(provider) {
    if (provider === 'groq') return state.groqToken;
    if (provider === 'openrouter') return state.openrouterToken;
    if (provider === 'together') return state.togetherToken;
    if (provider === 'deepinfra') return state.deepinfraToken;
    if (provider === 'fireworks') return state.fireworksToken;
    if (provider === 'baseten') return state.basetenToken;
    if (provider === 'nvidia') return state.nvidiaToken;
    return state.hfToken;
}

async function fetchModelApiAvailability(modelId, provider) {
    const cacheKey = `${provider}:${modelId}`;
    if (state.apiChecks[cacheKey]) {
        return state.apiChecks[cacheKey];
    }

    try {
        const res = await fetch('/models/api-check', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_id: modelId,
                provider,
                api_key: provider === 'hf' ? null : getProviderApiKey(provider),
                hf_token: state.hfToken || null
            })
        });
        const data = await res.json();
        state.apiChecks[cacheKey] = data;
        return data;
    } catch (e) {
        const fallback = {
            status: 'unknown',
            label: 'Sin verificar',
            mode: null,
            detail: 'No se pudo comprobar la API en este momento.'
        };
        state.apiChecks[cacheKey] = fallback;
        return fallback;
    }
}

async function enrichSearchResultApiStatus(models, provider) {
    const grid = document.getElementById('searchModelsGrid');
    if (!grid) return;

    for (const m of models) {
        const info = await fetchModelApiAvailability(m.id, provider);
        const card = Array.from(grid.children).find(node => {
            const idEl = node.querySelector('.m-id');
            return idEl && idEl.innerText === m.id;
        });
        if (!card) continue;
        const statusEl = card.querySelector('.m-api-status');
        if (!statusEl) continue;
        if (provider === 'fireworks' && info.status === 'unavailable') {
            card.remove();
            continue;
        }
        statusEl.className = `m-api-status ${info.status}`;
        const valueEl = statusEl.querySelector('.m-api-status-value');
        if (valueEl) valueEl.innerText = info.label;
    }
}

function showModelModal(m) {
    currentModalModel = m;
    const selectedProvider = state.searchProvider || state.provider || 'hf';
    const modal = document.getElementById('modelConfirmModal');
    document.getElementById('modalModelName').innerText = m.name;
    document.getElementById('modalModelId').innerText = m.id;
    document.getElementById('modalModelDesc').innerText = m.description || 'Sin descripción disponible.';
    document.getElementById('modalModelCapability').innerText = m.capability_label;
    
    const tags = document.getElementById('modalModelTags');
    tags.innerHTML = m.tags.map(t => `<span class="m-tag">${t}</span>`).join('');
    updateModelModalMeta(m, selectedProvider, state.apiChecks[`${selectedProvider}:${m.id}`] || null);
    
    document.getElementById('modalConfirmBtn').onclick = () => {
        setProvider(selectedProvider);
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
    const apiBtn = document.getElementById('modalConfirmBtn');
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

    apiBtn.disabled = true;
    apiBtn.innerText = 'Verificando API...';

    modal.style.display = 'flex';
    verifyModelApiAvailability(m, selectedProvider);
}

function getLocalStatusLabel(m) {
    if (m.is_downloaded) return 'Descargado';
    if (m.is_partial) return 'Descarga parcial';
    return 'No descargado';
}

function getRecommendedUsageLabel(m, apiInfo) {
    if (apiInfo && apiInfo.status === 'available') {
        return m.is_downloaded ? 'API o local' : 'API';
    }
    if (apiInfo && apiInfo.status === 'unavailable') {
        return m.is_downloaded ? 'Local' : 'Descargar local';
    }
    return m.is_downloaded ? 'Local o API sin verificar' : 'API sin verificar';
}

function updateModelModalMeta(m, provider, apiInfo) {
    const metaGrid = document.getElementById('modalMetaGrid');
    const apiBtn = document.getElementById('modalConfirmBtn');
    const metaItems = [];
    const addMeta = (label, value) => {
        if (value === undefined || value === null || value === '') return;
        metaItems.push(`
            <div class="modal-meta-card">
                <span class="modal-meta-label">${label}</span>
                <span class="modal-meta-value">${value}</span>
            </div>
        `);
    };

    addMeta('Task', m.pipeline_tag || 'chat');
    addMeta('Likes', typeof m.likes === 'number' ? m.likes.toLocaleString('es-AR') : null);
    addMeta('Descargas', typeof m.downloads === 'number' ? m.downloads.toLocaleString('es-AR') : null);
    addMeta('Soporte', m.supports_vision ? 'Vision + texto' : 'Solo texto');
    addMeta('Arquitectura', Array.isArray(m.architectures) && m.architectures.length ? m.architectures.join(', ') : null);
    addMeta('Tipo', m.model_type || null);
    addMeta('Proveedor elegido', provider.toUpperCase());
    addMeta('Estado local', getLocalStatusLabel(m));
    addMeta(
        provider === 'hf' ? 'API HF' : 'Estado API',
        apiInfo ? `${apiInfo.label}${apiInfo.mode ? ` (${apiInfo.mode})` : ''}` : 'Verificando...'
    );
    addMeta('Uso recomendado', getRecommendedUsageLabel(m, apiInfo));
    if (apiInfo && apiInfo.detail) {
        addMeta('Detalle API', apiInfo.detail);
    }
    metaGrid.innerHTML = metaItems.join('');

    if (!apiBtn) return;
    if (!apiInfo) {
        apiBtn.disabled = true;
        apiBtn.innerText = 'Verificando API...';
    } else if (apiInfo.status === 'unavailable') {
        apiBtn.disabled = true;
        apiBtn.innerText = 'API no disponible';
    } else {
        apiBtn.disabled = false;
        apiBtn.innerText = `Usar vía ${provider.toUpperCase()}`;
    }
}

async function verifyModelApiAvailability(m, provider) {
    const data = await fetchModelApiAvailability(m.id, provider);
    if (currentModalModel && currentModalModel.id === m.id) {
        updateModelModalMeta(m, provider, data);
    }
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

    let currentKey = state.hfToken;
    if (state.searchProvider === 'groq') currentKey = state.groqToken;
    if (state.searchProvider === 'openrouter') currentKey = state.openrouterToken;
    if (state.searchProvider === 'together') currentKey = state.togetherToken;
    if (state.searchProvider === 'deepinfra') currentKey = state.deepinfraToken;
    if (state.searchProvider === 'fireworks') currentKey = state.fireworksToken;
    if (state.searchProvider === 'baseten') currentKey = state.basetenToken;
    if (state.searchProvider === 'nvidia') currentKey = state.nvidiaToken;

    try {
        const res = await fetch('/models/search', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                query,
                limit: 10,
                task: document.getElementById('taskFilter').value,
                size_filter: document.getElementById('sizeFilter').value,
                provider: state.searchProvider,
                api_key: currentKey,
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

function listenModelProgress(modelId, mode = 'download') {
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

        if (modalOpen && modalId === modelId && mode === 'download') {
            if (dl.size_downloaded_gb !== undefined) {
                const pct = dl.progress || 0;
                document.getElementById('dlDownloadBar').style.width = pct + '%';
                document.getElementById('dlDownloadPct').innerText = pct + '%';
                document.getElementById('dlDownloadMsg').innerText = dl.message || '';
                const gb = (dl.size_downloaded_gb || 0).toFixed(2);
                document.getElementById('dlDownloadSize').innerText = `${gb} GB descargados`;
            }
            if (ld.status === 'loading' || ld.status === 'loaded') {
                document.getElementById('dlSectionLoad').style.display = 'block';
                const lpct = ld.progress || 0;
                document.getElementById('dlLoadBar').style.width = lpct + '%';
                document.getElementById('dlLoadPct').innerText = lpct + '%';
                document.getElementById('dlLoadMsg').innerText = ld.message || '';
            }
        }

        if (modalOpen && modalId === modelId && mode === 'load') {
            const lpct = ld.progress || 0;
            document.getElementById('dlLoadBar').style.width = lpct + '%';
            document.getElementById('dlLoadPct').innerText = lpct + '%';
            if (ld.message) document.getElementById('dlLoadMsg').innerText = ld.message;
        }

        if (mode === 'download' && dl.status === 'done') {
            delete state.downloading[modelId];
            loadLocalModels();
            loadFeaturedModels();
            if (modalOpen && modalId === modelId) {
                currentModalModel = { ...currentModalModel, is_downloaded: true, is_complete: true };
                showLoadStep(currentModalModel);
                showToast('✅ Descarga completa: ' + modelId.split('/').pop(), 'success');
            } else {
                showToast('✅ Descarga completa: ' + modelId.split('/').pop(), 'success');
            }
        }

        if (ld.status === 'loaded') {
            if (mode === 'load') {
                es.close();
                loadLocalModels();
                loadFeaturedModels();
                showToast('✅ Modelo cargado en memoria', 'success');
            }
            if (modalOpen && modalId === modelId) {
                document.getElementById('dlLoadMsg').innerText = '✓ Modelo cargado y listo';
                document.getElementById('dlLoadBar').style.width = '100%';
                document.getElementById('dlLoadPct').innerText = '100%';
                document.getElementById('dlModalActionsLoad').style.display = 'flex';
                const loadBtn = document.getElementById('dlLoadBtn');
                if (loadBtn) {
                    loadBtn.disabled = false;
                    loadBtn.innerText = '🧠 Cargar en memoria';
                }
            }
        }

        if (mode === 'download' && dl.status === 'error') {
            delete state.downloading[modelId];
            showToast('❌ Error en descarga: ' + (dl.message || ''), 'error');
            es.close();
        }
        if (mode === 'load' && ld.status === 'error') {
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

function listenDownloadProgress(modelId) {
    return listenModelProgress(modelId, 'download');
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
        listenLoadProgress(currentModalModel.id);
    } catch (e) {
        showToast(e.message, 'error');
        btn.disabled = false;
        btn.innerText = '🧠 Cargar en memoria';
    }
}

function listenLoadProgress(modelId) {
    return listenModelProgress(modelId, 'load');
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
    if (confirm("¿Limpiar historial de chat y documentos?")) {
        state.messages = [];
        clearPendingFiles();
        renderMessages();
        fetch('/rag/clear', { method: 'POST' }).catch(console.error);
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
