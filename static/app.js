/* ══════════════════════════════════════════════════════
   LLMFront – app.js v2
   Chat API + Descarga y ejecución local de modelos HF
   ══════════════════════════════════════════════════════ */

const API_BASE = window.location.origin;

// ── Estado global ─────────────────────────────────────
const state = {
  selectedModel: null,
  messages: [],
  isStreaming: false,
  hfToken: '',
  totalTokensEst: 0,
  mode: 'api',              // 'api' | 'local'
  loadedLocalModel: null,   // {id, quantization}
  quantization: 'none',

  config: {
    systemPrompt: 'Eres un asistente útil, amable y preciso. Respondes siempre en el idioma del usuario.',
    maxNewTokens: 512,
    temperature: 0.7,
    topP: 0.9,
    repPenalty: 1.1,
    stream: true,
  },
};

// Current download model data (shared between modals)
let pendingModelData = null;
let progressEventSource = null;
let systemVRAM = 0;
let systemRAMAvailable = 0;

// ── Init ───────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadFromStorage();
  setupTextarea();
  loadFeaturedModels();
  syncConfigUI();
  updateFooter();
  document.getElementById('newChatBtn').onclick = clearChat;
});

// ══════════════════════════════════════════════════════
// UTILS
// ══════════════════════════════════════════════════════

function showToast(msg, type = 'info', duration = 3000) {
  const toast = document.getElementById('toast');
  toast.textContent = msg;
  toast.className = `toast ${type} show`;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => toast.classList.remove('show'), duration);
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.appendChild(document.createTextNode(text));
  return div.innerHTML;
}

function formatMessageContent(text) {
  let html = escapeHtml(text);
  html = html.replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${code.trim()}</code></pre>`);
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\n/g, '<br>');
  return html;
}

function timeNow() {
  return new Date().toLocaleTimeString('es-AR', { hour: '2-digit', minute: '2-digit' });
}

function estimateTokens(text) {
  return Math.ceil(text.length / 4);
}

function updateStats() {
  document.getElementById('statMessages').textContent = state.messages.length;
  document.getElementById('statTokens').textContent = state.totalTokensEst.toLocaleString();
}

function formatNumber(n) {
  if (n >= 1000) return (n / 1000).toFixed(1) + 'k';
  return n;
}

// ══════════════════════════════════════════════════════
// LOCAL STORAGE
// ══════════════════════════════════════════════════════

function loadFromStorage() {
  try {
    const saved = JSON.parse(localStorage.getItem('llmfront_state') || '{}');
    if (saved.hfToken) {
      state.hfToken = saved.hfToken;
      document.getElementById('hfTokenInput').value = saved.hfToken;
      document.getElementById('hfTokenInput').classList.add('valid');
    }
    if (saved.config) Object.assign(state.config, saved.config);
    if (saved.selectedModel) {
      state.selectedModel = saved.selectedModel;
      updateModelDisplay();
    }
    if (saved.mode) {
      state.mode = saved.mode;
      updateModeUI();
    }
  } catch (e) {}
}

function saveToStorage() {
  try {
    localStorage.setItem('llmfront_state', JSON.stringify({
      hfToken: state.hfToken,
      config: state.config,
      selectedModel: state.selectedModel,
      mode: state.mode,
    }));
  } catch (e) {}
}

// ══════════════════════════════════════════════════════
// TOKEN
// ══════════════════════════════════════════════════════

function saveToken() {
  const input = document.getElementById('hfTokenInput');
  const val = input.value.trim();
  if (!val || !val.startsWith('hf_')) {
    showToast('El token debe comenzar con hf_', 'error'); return;
  }
  state.hfToken = val;
  input.classList.add('valid');
  saveToStorage();
  showToast('✓ Token guardado', 'success');
  updateSendButton();
}

// ══════════════════════════════════════════════════════
// MODE (API / LOCAL)
// ══════════════════════════════════════════════════════

function setMode(mode) {
  const prevMode = state.mode;
  state.mode = mode;
  updateModeUI();
  saveToStorage();
  updateSendButton();
  if (prevMode !== undefined && prevMode !== mode) {
    if (mode === 'local') showToast('Modo cambiado a Local 💻', 'info');
    else showToast('Modo cambiado a API ☁️', 'info');
  }
}

function updateModeUI() {
  const isLocal = state.mode === 'local';
  document.getElementById('modeApiBtn').classList.toggle('active', !isLocal);
  document.getElementById('modeLocalBtn').classList.toggle('active', isLocal);

  const localStatus = document.getElementById('localStatus');
  if (isLocal) {
    localStatus.style.display = 'flex';
    const lm = state.loadedLocalModel;
    document.getElementById('localStatusText').textContent = lm
      ? `${lm.id.split('/').pop()} cargado`
      : 'Sin modelo local cargado';
  } else {
    localStatus.style.display = 'none';
  }

  document.getElementById('footerMode').textContent = isLocal ? 'Local 💻' : 'API ☁️';
}

// ══════════════════════════════════════════════════════
// NAV / VIEWS
// ══════════════════════════════════════════════════════

function switchView(name) {
  ['chat', 'config', 'models'].forEach(v => {
    document.getElementById(`view${cap(v)}`).classList.remove('active');
    document.getElementById(`nav${cap(v)}`).classList.remove('active');
  });
  document.getElementById(`view${cap(name)}`).classList.add('active');
  document.getElementById(`nav${cap(name)}`).classList.add('active');
}

function cap(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

function toggleSidebar() {
  const sb = document.getElementById('sidebar');
  if (window.innerWidth <= 768) sb.classList.toggle('mobile-open');
  else sb.classList.toggle('collapsed');
}

// ══════════════════════════════════════════════════════
// MODELS TAB SWITCH
// ══════════════════════════════════════════════════════

function switchModelsTab(tab) {
  document.getElementById('tabCloud').classList.toggle('active', tab === 'cloud');
  document.getElementById('tabLocal').classList.toggle('active', tab === 'local');
  document.getElementById('tabContentCloud').style.display = tab === 'cloud' ? '' : 'none';
  document.getElementById('tabContentLocal').style.display = tab === 'local' ? '' : 'none';

  if (tab === 'local') {
    loadSystemInfo();
    loadLocalModels();
    loadFeaturedForDownload();
  }
}

// ══════════════════════════════════════════════════════
// SYSTEM INFO
// ══════════════════════════════════════════════════════

async function loadSystemInfo() {
  try {
    const r = await fetch(`${API_BASE}/system/info`);
    const d = await r.json();
    if (d.ram_total_gb) {
      systemRAMAvailable = d.ram_available_gb || 0;
      document.getElementById('hwRam').textContent =
        `🖥️ RAM: ${d.ram_available_gb} GB libre / ${d.ram_total_gb} GB total`;
    }
    const cudaEl = document.getElementById('hwCuda');
    if (d.cuda_available) {
      systemVRAM = d.vram_gb || 0;
      cudaEl.textContent = `⚡ CUDA: ${d.cuda_device || 'GPU'} (${systemVRAM} GB VRAM)`;
      cudaEl.style.color = 'var(--accent-green)';
    } else {
      systemVRAM = 0;
      cudaEl.textContent = '⚡ CUDA: No disponible (CPU mode)';
    }
  } catch (e) {
    document.getElementById('hwRam').textContent = '🖥️ RAM: —';
    systemVRAM = 0;
    systemRAMAvailable = 0;
  }
}

// ══════════════════════════════════════════════════════
// LOCAL MODELS LIST
// ══════════════════════════════════════════════════════

async function loadLocalModels() {
  const list = document.getElementById('localModelsList');
  try {
    const r = await fetch(`${API_BASE}/models/local`);
    const d = await r.json();
    document.getElementById('hwModels').textContent = `📦 Descargados: ${d.models.filter(m => m.is_complete).length}`;

    if (d.models.length === 0) {
      list.innerHTML = `<div class="local-empty">No hay modelos descargados aún.<br>
        Seleccioná un modelo de la lista de destacados y hacé clic en <strong>📥 Descargar local</strong>.</div>`;
      return;
    }

    list.innerHTML = d.models.map(m => {
      const isLoaded = state.loadedLocalModel?.id === m.id;

      // Ícono y badge según estado
      let icon, badge, actions;
      if (m.is_downloading) {
        icon = '⏬'; badge = '<span class="local-badge" style="background:rgba(255,157,0,.15);color:var(--accent-hf);border:1px solid rgba(255,157,0,.3)">Descargando</span>';
        actions = `<button class="local-action-btn delete" onclick="deleteLocalModel('${m.id}')" title="Cancelar descarga y borrar parciales">🛑 Cancelar</button>`;
      } else if (m.is_partial) {
        icon = '⚠️'; badge = `<span class="local-badge downloaded">${m.size_gb} GB parcial</span>`;
        actions = `
          <button class="local-action-btn load" onclick="resumeDownload('${m.id}',\`${m.name}\`)">🔄 Reanudar</button>
          <button class="local-action-btn delete" onclick="deleteLocalModel('${m.id}')" title="Eliminar archivos parciales">🗑️</button>`;
      } else if (isLoaded) {
        icon = '🟢'; badge = '<span class="local-badge loaded">CARGADO</span>';
        actions = `<button class="local-action-btn use" onclick="useLocalModelById('${m.id}')">✅ Usar</button>
                   <button class="local-action-btn delete" onclick="deleteLocalModel('${m.id}')">🗑️</button>`;
      } else {
        icon = '💿'; badge = '<span class="local-badge downloaded">LOCAL</span>';
        actions = `<button class="local-action-btn load" onclick="openLoadModal('${m.id}')">🧠 Cargar</button>
                   <button class="local-action-btn delete" onclick="deleteLocalModel('${m.id}')">🗑️</button>`;
      }

      return `
        <div class="local-model-card ${isLoaded ? 'is-loaded' : ''} ${m.is_partial ? 'is-partial' : ''}"
             id="lmc-${btoa(m.id).replace(/=/g,'')}">
          <div class="local-model-icon">${icon}</div>
          <div class="local-model-info">
            <div class="local-model-name">${m.name} ${badge}</div>
            <div class="local-model-meta">${m.id} · ${m.size_gb} GB${m.is_partial ? ' (incompleto)' : ''}</div>
          </div>
          <div class="local-model-actions">${actions}</div>
        </div>
      `;
    }).join('');
  } catch (e) {
    list.innerHTML = `<div class="local-empty">Error cargando lista local: ${e.message}</div>`;
  }
}

function resumeDownload(modelId, modelName) {
  const m = { id: modelId, name: modelName || modelId.split('/').pop() };
  pendingModelData = m;
  openDownloadModal(m, false);
  // El auto-check en openDownloadModal debería ver is_partial→downloading
  // Necesitamos disparar el download directamente:
  setTimeout(() => confirmDownload(), 300);
}


async function deleteLocalModel(modelId) {
  if (!confirm(`¿Eliminar "${modelId}" del disco? Esta acción no se puede deshacer.`)) return;
  try {
    const r = await fetch(`${API_BASE}/models/local/${encodeURIComponent(modelId)}`, { method: 'DELETE' });
    const d = await r.json();
    if (r.ok) {
      showToast('✓ Modelo eliminado', 'success');
      loadLocalModels();
      loadFeaturedForDownload();
    } else {
      showToast(`Error: ${d.detail}`, 'error');
    }
  } catch (e) {
    showToast(`Error: ${e.message}`, 'error');
  }
}

function openLoadModal(modelId) {
  // Open download modal showing load section
  const m = { id: modelId, name: modelId.split('/').pop() };
  pendingModelData = m;
  openDownloadModal(m, true);
}

function useLocalModelById(modelId) {
  state.loadedLocalModel = { id: modelId };
  state.selectedModel = { id: modelId, name: modelId.split('/').pop() };
  setMode('local');
  updateModelDisplay();
  saveToStorage();
  showToast(`✓ Usando ${modelId.split('/').pop()} en modo local`, 'success');
  switchView('chat');
}

// ══════════════════════════════════════════════════════
// FEATURED MODELS (Cloud tab)
// ══════════════════════════════════════════════════════

async function loadFeaturedModels() {
  const grid = document.getElementById('featuredModelsGrid');
  try {
    const res = await fetch(`${API_BASE}/models/featured`);
    const data = await res.json();
    renderModels(grid, data.models, 'api');
  } catch (e) {
    grid.innerHTML = `<div class="error-msg" style="grid-column:1/-1">⚠ ${e.message}</div>`;
  }
}

async function loadFeaturedForDownload() {
  const grid = document.getElementById('downloadModelsGrid');
  if (!grid) return;
  try {
    const res = await fetch(`${API_BASE}/models/featured`);
    const data = await res.json();
    renderModels(grid, data.models, 'local');
  } catch (e) {
    grid.innerHTML = `<div class="error-msg" style="grid-column:1/-1">⚠ ${e.message}</div>`;
  }
}

function renderModels(container, models, context = 'api') {
  container.innerHTML = models.map(m => {
    const isSelected = state.selectedModel?.id === m.id;
    const isDownloaded = m.is_downloaded || false;
    const isPartial = m.is_partial || false;
    const sizeTag = m.size_gb ? `<span class="model-size-tag">~${m.size_gb} GB</span>` : '';

    const dlBtn = context === 'local'
      ? `<div class="model-item-download">
          <button class="model-dl-btn ${isDownloaded ? 'downloaded' : ''} ${isPartial ? 'partial' : ''}"
            onclick='event.stopPropagation(); ${isDownloaded ? `openLoadModal("${m.id}")` : (isPartial ? `resumeDownload("${m.id}", "${m.name}")` : `openDownloadModal(${JSON.stringify(m)}, false)`)}'>
            ${isDownloaded ? '✅ Cargar en memoria' : (isPartial ? '🔄 Reanudar descarga' : '📥 Descargar local')}
          </button>
          ${sizeTag}
        </div>`
      : '';

    return `
      <div class="model-item ${isSelected ? 'selected' : ''}"
           data-model-id="${m.id}"
           onclick='openModelModal(${JSON.stringify(m)})'>
        <div class="model-item-header">
          <div>
            <div class="model-item-name">
              ${m.name || m.id.split('/').pop()}
              ${isDownloaded ? '<span class="local-badge downloaded" style="margin-left:4px">LOCAL</span>' : ''}
              ${isPartial ? '<span class="local-badge downloaded" style="margin-left:4px; background:rgba(255,157,0,.15); color:#ff9d00">PARCIAL</span>' : ''}
            </div>
            <div class="model-item-id">${m.id}</div>
          </div>
          <div class="model-likes">❤️ ${formatNumber(m.likes || 0)}</div>
        </div>
        ${m.description ? `<div class="model-item-desc">${m.description}</div>` : ''}
        <div class="model-tags">
          ${(m.tags || []).slice(0, 4).map(t => `<span class="model-tag">${t}</span>`).join('')}
        </div>
        ${dlBtn}
      </div>
    `;
  }).join('');
}

// ══════════════════════════════════════════════════════
// MODEL SEARCH
// ══════════════════════════════════════════════════════

let searchDebounceTimer = null;

function debounceSearch() {
  clearTimeout(searchDebounceTimer);
  const q = document.getElementById('modelSearchInput').value.trim();
  if (!q.length) { document.getElementById('searchResultsSection').style.display = 'none'; return; }
  searchDebounceTimer = setTimeout(searchModels, 600);
}

async function searchModels() {
  const q = document.getElementById('modelSearchInput').value.trim();
  const task = document.getElementById('taskFilter').value;
  const size = document.getElementById('sizeFilter').value;
  const grid = document.getElementById('searchModelsGrid');
  const section = document.getElementById('searchResultsSection');
  section.style.display = 'block';
  grid.innerHTML = '<div class="loading-spinner"><div class="spinner"></div><span>Buscando...</span></div>';
  try {
    const res = await fetch(`${API_BASE}/models/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: q, task, limit: 24, hf_token: state.hfToken || undefined, size_filter: size }),
    });
    const data = await res.json();
    if (!data.models.length) {
      grid.innerHTML = '<div style="grid-column:1/-1;color:var(--text-muted);padding:20px;font-size:13px;">Sin resultados.</div>';
    } else {
      renderModels(grid, data.models, 'api');
    }
  } catch (e) {
    grid.innerHTML = `<div class="error-msg" style="grid-column:1/-1">⚠ ${e.message}</div>`;
  }
}

// ══════════════════════════════════════════════════════
// MODEL SELECTION MODAL (API)
// ══════════════════════════════════════════════════════

function openModelModal(model) {
  pendingModelData = model;
  document.getElementById('modalModelName').textContent = model.name || model.id;
  document.getElementById('modalModelId').textContent = model.id;
  document.getElementById('modalModelTags').innerHTML = (model.tags || []).slice(0, 6)
    .map(t => `<span class="model-tag">${t}</span>`).join('');
  document.getElementById('modalModelDesc').textContent = model.description || 'Sin descripción disponible.';

  // Show size if available
  const dlBtn = document.getElementById('modalDownloadBtn');
  if (model.size_gb) dlBtn.textContent = `📥 Descargar local (~${model.size_gb} GB)`;
  else dlBtn.textContent = '📥 Descargar local';

  document.getElementById('modelConfirmModal').classList.add('open');
}

function closeModal() {
  document.getElementById('modelConfirmModal').classList.remove('open');
}

function confirmModelSelect() {
  if (!pendingModelData) return;
  state.selectedModel = pendingModelData;
  setMode('api');
  updateModelDisplay();
  saveToStorage();
  closeModal();
  showToast(`✓ Usando ${pendingModelData.name || pendingModelData.id} vía API`, 'success');
  updateSendButton();
  switchView('chat');
}

function startDownloadFromModal() {
  closeModal();
  openDownloadModal(pendingModelData, false);
}

function updateModelDisplay() {
  const m = state.selectedModel;
  if (!m) return;
  const name = m.name || m.id.split('/').pop();
  document.getElementById('activeModelName').textContent = name;
  document.getElementById('activeModelId').textContent = m.id;
  document.getElementById('modelCard').classList.add('has-model');
  document.getElementById('chatModelLabel').textContent = `${name} · ${m.id}`;
  document.getElementById('footerModel').textContent = m.id.split('/').pop();
  document.querySelectorAll('.model-item').forEach(el => {
    el.classList.toggle('selected', el.dataset.modelId === m.id);
  });
}

document.getElementById('changeModelBtn').onclick = () => switchView('models');
document.getElementById('modelConfirmModal').addEventListener('click', e => {
  if (e.target === document.getElementById('modelConfirmModal')) closeModal();
});

// ══════════════════════════════════════════════════════
// DOWNLOAD MODAL
// ══════════════════════════════════════════════════════

function openDownloadModal(model, skipToLoad = false) {
  pendingModelData = model;
  state.quantization = 'none';

  document.getElementById('dlModalModelId').textContent = model.id;
  document.getElementById('dlModalTitle').textContent = skipToLoad
    ? 'Cargar modelo en memoria'
    : `Descargar: ${model.name || model.id.split('/').pop()}`;

  // Reset UI
  setProgress('download', 0, 'Listo para iniciar');
  setProgress('load', 0, 'Esperando...');
  document.getElementById('dlDownloadSize').textContent = '';
  document.getElementById('dlSectionDownload').style.display = skipToLoad ? 'none' : '';
  document.getElementById('dlSectionLoad').style.display = skipToLoad ? '' : 'none';
  
  // Restore original buttons in case they were overwritten by a completed download msg
  document.getElementById('dlModalActions').innerHTML = `
      <button onclick="closeDownloadModal()" class="modal-cancel-btn">Cerrar</button>
      <button onclick="confirmDownload()" class="modal-confirm-btn" id="dlConfirmBtn">
        📥 Iniciar descarga
      </button>
  `;
  document.getElementById('dlModalActions').style.display = skipToLoad ? 'none' : '';
  document.getElementById('dlModalActionsLoad').style.display = skipToLoad ? '' : 'none';
  document.getElementById('dlUseBtn').style.display = 'none';
  document.getElementById('dlLoadBtn').style.display = '';

  let recommended = 'none';
  const size = model.size_gb || 0;
  const hintEl = document.getElementById('dlQuantHint');
  
  if (size > 0) {
      const sizeNone = size + 1.0;
      const size8bit = size * 0.55 + 1.0;
      const size4bit = size * 0.3 + 1.0;

      document.getElementById('qNone').textContent = `🔵 Completo (~${sizeNone.toFixed(1)} GB)`;
      document.getElementById('q8bit').textContent = `🟡 8-bit (~${size8bit.toFixed(1)} GB)`;
      document.getElementById('q4bit').textContent = `🟢 4-bit (~${size4bit.toFixed(1)} GB)`;

      if (systemVRAM > 0) {
          if (sizeNone > systemVRAM) {
              if (size8bit > systemVRAM) {
                  recommended = '4bit';
                  hintEl.innerHTML = `<span style="color:#ef4444; font-weight: 500;">⚠️ PELIGRO OOM: El modelo (~${size.toFixed(1)} GB) excede tu VRAM (${systemVRAM.toFixed(1)} GB). <strong>Debés usar 4-bit</strong> o crasheara.</span>`;
              } else {
                  recommended = '8bit';
                  hintEl.innerHTML = `<span style="color:#ff9d00; font-weight: 500;">⚠️ ADVERTENCIA: Usá 8-bit o 4-bit para que entre en tu VRAM (${systemVRAM.toFixed(1)} GB).</span>`;
              }
          } else {
              recommended = 'none';
              hintEl.innerHTML = `<span style="color:#4ade80; font-weight: 500;">✅ El modelo en versión Completa (~${sizeNone.toFixed(1)} GB) entrará cómodamente en tus ${systemVRAM.toFixed(1)} GB de VRAM.</span>`;
          }
      } else {
         if (hintEl) hintEl.textContent = `4-bit y 8-bit requieren CUDA + bitsandbytes instalado.`;
      }
  } else {
      document.getElementById('qNone').textContent = `🔵 Completo`;
      document.getElementById('q8bit').textContent = `🟡 8-bit`;
      document.getElementById('q4bit').textContent = `🟢 4-bit (CUDA)`;
      if (hintEl) hintEl.textContent = `4-bit y 8-bit requieren CUDA + bitsandbytes instalado. Reducen VRAM a costa de calidad.`;
      recommended = 'none';
  }

  setQuant(recommended);
  document.getElementById('downloadModal').classList.add('open');

  // Auto-verificar si ya está descargado (por si el SSE cayó)
  if (!skipToLoad) {
    fetch(`${API_BASE}/models/download/${encodeURIComponent(model.id)}/status`)
      .then(r => r.json())
      .then(status => {
        if (status.is_downloaded) {
          // Ya completó: mostrar sección de carga directamente
          setProgress('download', 100, '✓ Ya descargado');
          if (status.download?.size_gb) {
            document.getElementById('dlDownloadSize').textContent = `${status.download.size_gb} GB en disco`;
          }
          document.getElementById('dlSectionLoad').style.display = '';
          document.getElementById('dlModalActions').style.display = 'none';
          document.getElementById('dlModalActionsLoad').style.display = '';
          if (status.is_loaded) {
            setProgress('load', 100, '✓ Modelo en memoria');
            document.getElementById('dlLoadBtn').style.display = 'none';
            document.getElementById('dlUseBtn').style.display = '';
          }
        } else if (status.download?.status === 'downloading') {
          // Sigue descargando: iniciar seguimiento
          startProgressStream(model.id);
        }
      })
      .catch(() => {}); // Si falla, el usuario hace clic manual
  }
}


function closeDownloadModal() {
  document.getElementById('downloadModal').classList.remove('open');
  if (progressEventSource) { progressEventSource.close(); progressEventSource = null; }
}

document.getElementById('downloadModal').addEventListener('click', e => {
  if (e.target === document.getElementById('downloadModal')) closeDownloadModal();
});

function setQuant(q) {
  state.quantization = q;
  ['qNone', 'q8bit', 'q4bit'].forEach(id => document.getElementById(id).classList.remove('active'));
  const map = { none: 'qNone', '8bit': 'q8bit', '4bit': 'q4bit' };
  document.getElementById(map[q]).classList.add('active');
}

function setProgress(section, pct, msg) {
  const id = section === 'download' ? 'Download' : 'Load';
  const bar = document.getElementById(`dl${id}Bar`);
  const pctEl = document.getElementById(`dl${id}Pct`);
  const msgEl = document.getElementById(`dl${id}Msg`);
  if (bar) bar.style.width = `${Math.min(pct, 100)}%`;
  if (pctEl) pctEl.textContent = `${Math.round(pct)}%`;
  if (msgEl) msgEl.textContent = msg;
}

async function confirmDownload() {
  if (!pendingModelData) return;

  const btn = document.getElementById('dlConfirmBtn');
  btn.disabled = true;
  btn.textContent = '⏳ Iniciando...';

  try {
    const r = await fetch(`${API_BASE}/models/download`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_id: pendingModelData.id,
        hf_token: state.hfToken || undefined,
        quantization: state.quantization,
      }),
    });
    const d = await r.json();
    if (d.status === 'already_downloading') {
      showToast('Ya se está descargando este modelo', 'info');
    }
    startProgressStream(pendingModelData.id);
  } catch (e) {
    showToast(`Error: ${e.message}`, 'error');
    btn.disabled = false;
    btn.textContent = '📥 Iniciar descarga';
  }
}

function startProgressStream(modelId) {
  if (progressEventSource) { progressEventSource.close(); progressEventSource = null; }

  // Barra indeterminada: ancho fijo al 100%, la animación shimmer da sensación de activo.
  // Solo ponemos el % real cuando lo conocemos (done = 100%).
  let downloadDone = false;

  // Pulsar la barra para indicar actividad (no fake %)
  const bar = document.getElementById('dlDownloadBar');
  const pctEl = document.getElementById('dlDownloadPct');
  if (bar) { bar.style.width = '100%'; bar.classList.add('dl-bar-indeterminate'); }
  if (pctEl) pctEl.textContent = '⏳';
  setProgress('download', 0, 'Iniciando descarga...');
  // Resetear al 100% indeterminado
  if (bar) bar.style.width = '60%'; // Valor visible amplio

  // ── Polling de respaldo cada 3s ──
  const pollInterval = setInterval(async () => {
    if (downloadDone) { clearInterval(pollInterval); return; }
    try {
      const r = await fetch(`${API_BASE}/models/download/${encodeURIComponent(modelId)}/status`);
      const status = await r.json();
      handleStatusUpdate(status.download || {}, status.load || {});
    } catch (_) {}
  }, 3000);

  // ── Procesamiento central ──
  function handleStatusUpdate(dl, ld) {
    // Mostrar GB descargados en tiempo real
    if (dl.size_downloaded_gb != null) {
      const gb = typeof dl.size_downloaded_gb === 'number' ? dl.size_downloaded_gb.toFixed(2) : dl.size_downloaded_gb;
      const saved = dl.size_saved_gb ? ` (+${dl.size_saved_gb} GB previos)` : '';
      document.getElementById('dlDownloadSize').textContent = `${gb} GB descargados${saved}`;
      document.getElementById('dlDownloadMsg').textContent = dl.message || 'Descargando...';
    }
    if (dl.status === 'downloading') {
      const btn = document.getElementById('dlConfirmBtn');
      if (btn) {
        btn.disabled = true;
        btn.textContent = '⏬ Descargando...';
      }
    }

    if ((dl.status === 'done' || dl.status === 'partial') && !downloadDone) {
      const isDone = dl.status === 'done';
      downloadDone = isDone; // solo marcar completo si realmente terminó
      clearInterval(pollInterval);

      if (isDone) {
        if (bar) { bar.style.width = '100%'; bar.classList.remove('dl-bar-indeterminate'); }
        if (pctEl) pctEl.textContent = '100%';
        document.getElementById('dlDownloadMsg').textContent = dl.message || '✓ Descarga completada';
        if (dl.size_gb) document.getElementById('dlDownloadSize').textContent = `${dl.size_gb} GB en disco`;

        // Ocultamos las acciones de descarga y solo mostramos el botón de cerrar
        document.getElementById('dlModalActions').style.display = 'none';
        document.getElementById('dlModalActionsLoad').style.display = 'none';
        
        const actionsDiv = document.getElementById('dlModalActions');
        actionsDiv.style.display = '';
        actionsDiv.innerHTML = `<button onclick="closeDownloadModal()" class="modal-confirm-btn" style="width: 100%;">✓ Listo, cerrar</button>`;
        
        showToast('✓ Descarga completa. Podes cargarlo manualmente más tarde.', 'success');
        loadLocalModels();
        loadFeaturedForDownload();
      } else {
        // partial: descarga interrumpida, se puede reintentar
        if (bar) { bar.style.width = '30%'; bar.classList.remove('dl-bar-indeterminate'); }
        if (pctEl) pctEl.textContent = '—';
        document.getElementById('dlDownloadMsg').textContent = dl.message || 'Interrumpida';
        // Reactivar el botón de iniciar descarga para reintentar
        const btn = document.getElementById('dlConfirmBtn');
        if (btn) {
          btn.disabled = false;
          btn.textContent = '🔄 Reanudar descarga';
          document.getElementById('dlModalActions').style.display = '';
        }
        loadLocalModels();
      }

    } else if (dl.status === 'error' && !downloadDone) {
      clearInterval(pollInterval);
      if (bar) { bar.style.width = '0%'; bar.classList.remove('dl-bar-indeterminate'); }
      if (pctEl) pctEl.textContent = '!';
      document.getElementById('dlDownloadMsg').textContent = `Error: ${dl.message}`;
      const btn = document.getElementById('dlConfirmBtn');
      if (btn) { btn.disabled = false; btn.textContent = '🔄 Reintentar'; }
      showToast(`Error: ${dl.message}`, 'error', 6000);
    }

    // Sección de carga en memoria
    if (ld.status === 'loading') {
      const loadBar = document.getElementById('dlLoadBar');
      if (loadBar) { loadBar.style.width = '60%'; loadBar.classList.add('dl-bar-indeterminate'); }
      document.getElementById('dlLoadPct').textContent = '⏳';
      document.getElementById('dlLoadMsg').textContent = ld.message || 'Cargando modelo...';
    } else if (ld.status === 'loaded') {
      const loadBar = document.getElementById('dlLoadBar');
      if (loadBar) { loadBar.style.width = '100%'; loadBar.classList.remove('dl-bar-indeterminate'); }
      document.getElementById('dlLoadPct').textContent = '100%';
      document.getElementById('dlLoadMsg').textContent = ld.message || '✓ Modelo cargado';
      document.getElementById('dlLoadBtn').style.display = 'none';
      document.getElementById('dlUseBtn').style.display = '';
      state.loadedLocalModel = { id: modelId, quantization: state.quantization };
      updateModeUI();
      loadLocalModels();
      showToast('✓ Modelo listo para usar', 'success', 5000);
    } else if (ld.status === 'error') {
      document.getElementById('dlLoadMsg').textContent = `Error: ${ld.message}`;
      document.getElementById('dlLoadPct').textContent = '!';
      showToast(`Error al cargar: ${ld.message}`, 'error', 8000);
    }
  }

  // ── SSE (principal) ──
  try {
    progressEventSource = new EventSource(
      `${API_BASE}/models/download/${encodeURIComponent(modelId)}/progress`
    );
    progressEventSource.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data._end) { progressEventSource.close(); progressEventSource = null; return; }
        handleStatusUpdate(data.download || {}, data.load || {});
      } catch (_) {}
    };
    progressEventSource.onerror = () => {
      if (progressEventSource) { progressEventSource.close(); progressEventSource = null; }
      // El polling de respaldo sigue activo
    };
  } catch (_) {}
}



async function loadLocalModel() {
  if (!pendingModelData) return;

  const size = pendingModelData.size_gb || 0;
  if (size > 0) {
    let sizeAdjusted = size;
    if (state.quantization === '8bit') sizeAdjusted = size / 2;
    else if (state.quantization === '4bit') sizeAdjusted = size / 3;

    if (systemVRAM > 0) {
      if (sizeAdjusted + 1.5 > systemVRAM) {
        if (!confirm(`⚠️ ADVERTENCIA DE HARDWARE\n\nEl modelo requiere un estimado de ${(sizeAdjusted + 1.5).toFixed(1)} GB de VRAM, pero tu sistema reporta ${systemVRAM.toFixed(1)} GB de VRAM total.\n\nEsto puede causar un error de falta de memoria (OOM) y el servidor podría crashear.\n\n¿Deseas intentar cargarlo de todos modos?`)) return;
      }
    } else if (systemRAMAvailable > 0) {
      if (sizeAdjusted + 1.5 > systemRAMAvailable) {
        if (!confirm(`⚠️ ADVERTENCIA DE HARDWARE\n\nEl modelo requiere un estimado de ${(sizeAdjusted + 1.5).toFixed(1)} GB de RAM, pero tu sistema solo tiene ${systemRAMAvailable.toFixed(1)} GB libres.\n\nEsto puede causar que el sistema se quede sin memoria (OOM).\n\n¿Deseas intentar cargarlo de todos modos?`)) return;
      }
    }
  }

  const btn = document.getElementById('dlLoadBtn');
  btn.disabled = true;
  btn.textContent = '⏳ Cargando...';

  try {
    const r = await fetch(`${API_BASE}/models/load`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_id: pendingModelData.id,
        quantization: state.quantization,
        device: 'auto',
      }),
    });
    const d = await r.json();
    if (d.status === 'already_loaded') {
      showToast('Modelo ya está en memoria', 'info');
      state.loadedLocalModel = { id: pendingModelData.id };
      document.getElementById('dlLoadBtn').style.display = 'none';
      document.getElementById('dlUseBtn').style.display = '';
      return;
    }
    setProgress('load', 5, 'Iniciando carga...');
    document.getElementById('dlSectionLoad').style.display = '';
    startProgressStream(pendingModelData.id);
  } catch (e) {
    showToast(`Error: ${e.message}`, 'error');
    btn.disabled = false;
    btn.textContent = '🧠 Cargar en memoria';
  }
}

function useLocalModel() {
  if (!pendingModelData) return;
  state.selectedModel = pendingModelData;
  state.loadedLocalModel = { id: pendingModelData.id, quantization: state.quantization };
  setMode('local');
  updateModelDisplay();
  saveToStorage();
  closeDownloadModal();
  showToast(`✓ ${pendingModelData.name || pendingModelData.id.split('/').pop()} listo!`, 'success');
  updateSendButton();
  switchView('chat');
}

// ══════════════════════════════════════════════════════
// CHAT
// ══════════════════════════════════════════════════════

function setupTextarea() {
  const ta = document.getElementById('userInput');
  ta.addEventListener('input', () => {
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 200) + 'px';
    document.getElementById('charCount').textContent = `${ta.value.length} / 8000`;
    updateSendButton();
  });
  ta.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!document.getElementById('sendBtn').disabled) sendMessage();
    }
  });
}

function updateSendButton() {
  const ta = document.getElementById('userInput');
  const btn = document.getElementById('sendBtn');
  const hasModel = !!state.selectedModel;
  const hasText = !!ta.value.trim();
  const localOk = state.mode !== 'local' || !!state.loadedLocalModel;
  btn.disabled = !hasModel || !hasText || state.isStreaming || !localOk;
}

async function sendMessage() {
  const ta = document.getElementById('userInput');
  const text = ta.value.trim();
  if (!text || !state.selectedModel || state.isStreaming) return;

  document.getElementById('welcomeScreen')?.remove();
  addMessage('user', text);
  state.messages.push({ role: 'user', content: text });
  state.totalTokensEst += estimateTokens(text);
  ta.value = '';
  ta.style.height = 'auto';
  document.getElementById('charCount').textContent = '0 / 8000';
  updateSendButton();
  updateStats();
  syncConfigFromUI();
  await streamResponse();
}

function addMessage(role, content) {
  const msgs = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = `message ${role}`;
  const avatar = role === 'user' ? '👤' : '🤗';
  div.innerHTML = `
    <div class="msg-avatar">${avatar}</div>
    <div class="msg-content">
      <div class="msg-bubble" data-content></div>
      <div class="msg-time">${timeNow()}</div>
    </div>
  `;
  div.querySelector('[data-content]').innerHTML = formatMessageContent(content);
  msgs.appendChild(div);
  scrollToBottom();
  return div;
}

function scrollToBottom() {
  const wrap = document.getElementById('messagesWrap');
  wrap.scrollTop = wrap.scrollHeight;
}

async function streamResponse() {
  state.isStreaming = true;
  updateSendButton();

  const msgs = document.getElementById('messages');
  const typingEl = document.createElement('div');
  typingEl.className = 'message assistant';
  typingEl.id = 'typingIndicator';
  typingEl.innerHTML = `
    <div class="msg-avatar">🤗</div>
    <div class="typing-indicator">
      <div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>
    </div>
  `;
  msgs.appendChild(typingEl);
  scrollToBottom();

  const payload = buildPayload();

  try {
    const res = await fetch(`${API_BASE}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    typingEl.remove();
    const msgEl = addMessage('assistant', '');
    const bubble = msgEl.querySelector('[data-content]');
    let fullText = '';

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const data = JSON.parse(line.slice(6).trim());
          if (data.token) {
            fullText += data.token;
            bubble.innerHTML = formatMessageContent(fullText);
            scrollToBottom();
          } else if (data.error) {
            bubble.innerHTML = `<div class="error-msg">⚠ ${data.error}</div>`;
          } else if (data.done) {
            state.messages.push({ role: 'assistant', content: fullText });
            state.totalTokensEst += estimateTokens(fullText);
            updateStats();
          }
        } catch (_) {}
      }
    }
  } catch (err) {
    typingEl.remove();
    const msgEl = addMessage('assistant', '');
    msgEl.querySelector('[data-content]').innerHTML =
      `<div class="error-msg">⚠ Error de conexión: ${err.message}</div>`;
  } finally {
    state.isStreaming = false;
    updateSendButton();
  }
}

function buildPayload() {
  const cfg = state.config;
  const useLocal = state.mode === 'local' && !!state.loadedLocalModel;
  return {
    model: useLocal ? state.loadedLocalModel.id : state.selectedModel.id,
    messages: state.messages.filter(m => m.role !== 'system'),
    system_prompt: cfg.systemPrompt,
    max_new_tokens: cfg.maxNewTokens,
    temperature: cfg.temperature,
    top_p: cfg.topP,
    repetition_penalty: cfg.repPenalty,
    stream: true,
    hf_token: state.hfToken || undefined,
    use_local: useLocal,
  };
}

function clearChat() {
  state.messages = [];
  state.totalTokensEst = 0;
  updateStats();
  const msgs = document.getElementById('messages');
  msgs.innerHTML = `
    <div class="welcome-screen" id="welcomeScreen">
      <div class="welcome-icon">🤗</div>
      <h1 class="welcome-title">Chat limpio</h1>
      <p class="welcome-sub">El historial fue borrado. ¡Empezá una nueva conversación!</p>
    </div>
  `;
  showToast('Chat limpiado', 'info');
}

function exportChat() {
  if (!state.messages.length) { showToast('No hay mensajes para exportar', 'error'); return; }
  const model = state.selectedModel?.id || 'unknown';
  const lines = [`# Chat Export – ${model}\nFecha: ${new Date().toLocaleString('es-AR')}\nModo: ${state.mode}\n\n`];
  state.messages.forEach(m => {
    const role = m.role === 'user' ? '**Tú**' : `**${model.split('/').pop()}**`;
    lines.push(`${role}:\n${m.content}\n\n---\n\n`);
  });
  const blob = new Blob([lines.join('')], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `chat_${Date.now()}.md`;
  a.click();
  URL.revokeObjectURL(url);
  showToast('Chat exportado', 'success');
}

// ══════════════════════════════════════════════════════
// CONFIG
// ══════════════════════════════════════════════════════

function syncConfigUI() {
  const cfg = state.config;
  document.getElementById('systemPrompt').value = cfg.systemPrompt;
  document.getElementById('temperature').value = cfg.temperature;
  document.getElementById('tempValue').textContent = cfg.temperature;
  document.getElementById('maxNewTokens').value = cfg.maxNewTokens;
  document.getElementById('maxTokensValue').textContent = cfg.maxNewTokens;
  document.getElementById('topP').value = cfg.topP;
  document.getElementById('topPValue').textContent = cfg.topP;
  document.getElementById('repPenalty').value = cfg.repPenalty;
  document.getElementById('repPenValue').textContent = cfg.repPenalty;
  document.getElementById('streamToggle').checked = cfg.stream;
  updateFooter();
}

function syncConfigFromUI() {
  state.config.systemPrompt = document.getElementById('systemPrompt').value;
  state.config.temperature = parseFloat(document.getElementById('temperature').value);
  state.config.maxNewTokens = parseInt(document.getElementById('maxNewTokens').value);
  state.config.topP = parseFloat(document.getElementById('topP').value);
  state.config.repPenalty = parseFloat(document.getElementById('repPenalty').value);
  state.config.stream = document.getElementById('streamToggle').checked;
  saveToStorage();
  updateFooter();
}

function updateSlider(sliderId, displayId, val) {
  document.getElementById(displayId).textContent = parseFloat(val);
  updateFooter();
}

function updateFooter() {
  document.getElementById('footerTokens').textContent = document.getElementById('maxNewTokens')?.value || state.config.maxNewTokens;
  document.getElementById('footerTemp').textContent = document.getElementById('temperature')?.value || state.config.temperature;
}

const PRESETS = {
  precise:  { temperature: 0.2,  topP: 0.85, repPenalty: 1.05, maxNewTokens: 512,  systemPrompt: 'Eres un asistente preciso y factual. Da respuestas directas, correctas y concisas.' },
  balanced: { temperature: 0.7,  topP: 0.9,  repPenalty: 1.1,  maxNewTokens: 512,  systemPrompt: 'Eres un asistente útil, amable y preciso. Respondes siempre en el idioma del usuario.' },
  creative: { temperature: 1.2,  topP: 0.95, repPenalty: 1.2,  maxNewTokens: 1024, systemPrompt: 'Eres un asistente creativo e imaginativo. ¡Sorpréndeme!' },
  code:     { temperature: 0.3,  topP: 0.85, repPenalty: 1.05, maxNewTokens: 2048, systemPrompt: 'Eres un experto en programación. Respondes con código limpio y bien documentado. Siempre usás bloques de código markdown.' },
};

function applyPreset(name) {
  Object.assign(state.config, PRESETS[name]);
  syncConfigUI();
  saveToStorage();
  showToast(`✓ Preset "${name}" aplicado`, 'success');
}

function resetConfig() {
  Object.assign(state.config, PRESETS.balanced);
  syncConfigUI();
  saveToStorage();
  showToast('Configuración restaurada', 'info');
}
