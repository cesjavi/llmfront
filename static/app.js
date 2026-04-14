// LLMFront - Frontend logic v2.1
let state = {
    view: 'chat',
    modelsTab: 'cloud',
    activeModel: null, // {id, name, isLocal, supportsVision}
    isLocal: false,
    messages: [],
    pendingImages: [], // [{base64, mimeType}]
    hfToken: localStorage.getItem('hf_token') || '',
    systemInfo: null,
    downloading: {},
    quant: 'none',
    cameraStream: null
};

// ─── INIT ───
document.addEventListener('DOMContentLoaded', () => {
    if (state.hfToken) document.getElementById('hfTokenInput').value = state.hfToken;
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

    const payload = {
        model: state.activeModel.id,
        messages: state.messages.slice(0, -1).map(m => ({ role: m.role, content: m.content })),
        system_prompt: document.getElementById('systemPrompt').value,
        temperature: parseFloat(document.getElementById('temperature').value),
        max_new_tokens: parseInt(document.getElementById('maxNewTokens').value),
        top_p: parseFloat(document.getElementById('topP').value),
        repetition_penalty: parseFloat(document.getElementById('repPenalty').value),
        stream: document.getElementById('streamToggle').checked,
        hf_token: state.hfToken,
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
        const decoder = new TextDecoder();
        let aiFullText = "";
        aiMsg.thinking = false;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.substring(6));
                        if (data.token) {
                            aiFullText += data.token;
                            aiMsg.content = aiFullText;
                            renderMessages();
                            scroll();
                        }
                        if (data.error) throw new Error(data.error);
                    } catch (e) { console.error("JSON parse error", e); }
                }
            }
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
        container.appendChild(document.getElementById('welcomeScreen'));
        return;
    }

    state.messages.forEach((msg, idx) => {
        const div = document.createElement('div');
        div.className = `message ${msg.role === 'user' ? 'user-msg' : 'ai-msg'}`;
        
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
        
        contentHtml += `<div class="msg-text">${mdText}</div>`;
        div.innerHTML = contentHtml;
        container.appendChild(div);
    });
}

function formatMarkdown(text) {
    // Basic formatting for demo
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\n/g, '<br/>');
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

function setMode(mode) {
    state.isLocal = (mode === 'local');
    document.getElementById('modeApiBtn').classList.toggle('active', mode === 'api');
    document.getElementById('modeLocalBtn').classList.toggle('active', mode === 'local');
    
    document.getElementById('footerMode').innerText = mode === 'api' ? 'API ☁️' : 'Local 💻';
    updateVisionSupport();
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
        renderModels(data.models, 'featuredModelsGrid');
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

function renderModels(models, gridId) {
    const grid = document.getElementById(gridId);
    grid.innerHTML = '';
    models.forEach(m => {
        const card = document.createElement('div');
        card.className = `model-card-item ${state.activeModel?.id === m.id ? 'active' : ''}`;
        card.onclick = () => showModelModal(m);
        card.innerHTML = `
            <div class="m-header">
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
        // The SSE will handle progress automatically
    } catch (e) { showToast(e.message, "error"); }
}

async function loadLocalModel() {
    const btn = document.getElementById('dlLoadBtn');
    btn.disabled = true;
    btn.innerText = "⏳ Cargando...";
    
    document.getElementById('dlSectionLoad').style.display = 'block';

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
        showToast("Iniciando carga...", "info");
    } catch (e) { showToast(e.message, "error"); }
}

function useLocalModel() {
    setActiveModel(currentModalModel, true);
    closeDownloadModal();
    switchView('chat');
}

function closeDownloadModal() {
    document.getElementById('downloadModal').style.display = 'none';
}

function checkDownloadsProgress() {
    // This could call /models/download/{id}/status for active downloads
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
