/* ═══════════════════════════════════════════════════════════════
   WinneAI — Client-side JavaScript
   Chat, Sessions, Settings, Navigation
   ═══════════════════════════════════════════════════════════════ */

// ── State ──
let currentPage = 'chat';
let activeSessionId = null;
let sessions = [];
let isSending = false;
let sessionsPanelOpen = false;
let activeJobId = null;

// ── Settings defaults ──
const settings = {
    topK: 10,
    threshold: 0.30,
    rerankTopK: 3,
    maxContext: 10000,
};

// ══════════════════════════════════════════════════════════════
// Init
// ══════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', async () => {
    await loadStatus();
    // Don't create a server session yet - use a temp ID
    activeSessionId = 'temp-' + Date.now();
    await loadSessions();
    clearChatUI();
    updateSendButton();
});

// ══════════════════════════════════════════════════════════════
// Navigation
// ══════════════════════════════════════════════════════════════

function navigateTo(page) {
    currentPage = page;
    document.querySelectorAll('#page-chat, #page-library, #page-models, #page-settings').forEach(el => el.classList.add('hidden'));
    document.getElementById(`page-${page}`).classList.remove('hidden');

    // Update nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        const p = link.dataset.page;
        if (p === page) {
            link.className = 'nav-link flex items-center gap-3 px-4 py-3 bg-white text-primary rounded-xl shadow-sm font-semibold transition-all duration-200 cursor-pointer';
            link.querySelector('.material-symbols-outlined').style.fontVariationSettings = "'FILL' 1";
        } else {
            link.className = 'nav-link flex items-center gap-3 px-4 py-3 text-on-surface-variant hover:bg-surface-container-high rounded-xl transition-all duration-200 cursor-pointer';
            link.querySelector('.material-symbols-outlined').style.fontVariationSettings = "'FILL' 0";
        }
    });

    // Chat 외 페이지에서는 세션 패널 숨기기, Chat 복귀 시 복원
    const toggleBtn = document.getElementById('sessions-toggle-btn');
    const panel = document.getElementById('sessions-panel');
    if (page !== 'chat') {
        if (sessionsPanelOpen) {
            panel.classList.add('collapsed');
        }
        toggleBtn.classList.add('hidden');
    } else {
        toggleBtn.classList.remove('hidden');
        if (sessionsPanelOpen) {
            panel.classList.remove('collapsed');
            loadSessions();
        }
    }

    if (page === 'settings') { loadStatus(); loadConfig(); }
}

// ══════════════════════════════════════════════════════════════
// Status
// ══════════════════════════════════════════════════════════════

async function loadStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        const badge = document.getElementById('status-badge');
        if (data.busy) {
            const sec = Math.round(data.busy_for_sec || 0);
            badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-amber-500 animate-pulse"></span>
                <span class="text-amber-600">Model busy · ${sec}s</span>`;
        } else if (data.ready) {
            badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-success"></span>
                <span class="text-success">Model: ${data.model_name} · Ready</span>`;
        } else {
            badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-outline animate-pulse"></span>
                <span class="text-outline">No index</span>`;
        }
        document.getElementById('model-tag').textContent = data.model_name;
        // Settings info cards
        document.getElementById('info-llm').textContent = data.model_name;
        document.getElementById('info-embed').textContent = data.embedding_model;
        document.getElementById('info-index').textContent = `${data.chunk_count} chunks`;
    } catch (e) {
        console.error('Status load failed:', e);
    }
}

// ══════════════════════════════════════════════════════════════
// Sessions
// ══════════════════════════════════════════════════════════════

async function createSession() {
    activeSessionId = 'temp-' + Date.now();
    clearChatUI();
    renderSessionList();
    renderSessionPanel();
}

async function loadSessions() {
    try {
        const res = await fetch('/api/sessions');
        sessions = await res.json();
        renderSessionList();
        renderSessionPanel();
    } catch (e) {
        console.error('Load sessions failed:', e);
    }
}

function renderSessionList() {
    // Sidebar session list removed — only Sessions panel used now
}

async function switchSession(sessionId) {
    activeSessionId = sessionId;
    renderSessionList();
    renderSessionPanel();

    // Load session messages
    try {
        const res = await fetch(`/api/sessions/${sessionId}`);
        const session = await res.json();
        clearChatUI();
        if (session.messages && session.messages.length > 0) {
            document.getElementById('empty-state').classList.add('hidden');
            session.messages.forEach(msg => {
                if (msg.role === 'user') {
                    appendUserMessage(msg.content);
                } else if (msg.role === 'assistant') {
                    appendAIMessage(msg.content, msg.evidences || [], msg.reference_materials || {}, msg.llm_sec || 0);
                }
            });
        }
    } catch (e) {
        console.error('Switch session failed:', e);
    }
    navigateTo('chat');
}

// ── Confirm dialog ──
let _pendingDeleteId = null;

function confirmDeleteSession(sessionId) {
    _pendingDeleteId = sessionId;
    const s = sessions.find(x => x.id === sessionId);
    const title = s ? s.title : '이 세션';
    document.getElementById('confirm-msg').textContent = `"${title}" 세션을 삭제하시겠습니까?`;
    const overlay = document.getElementById('confirm-overlay');
    overlay.classList.remove('opacity-0', 'pointer-events-none');
    overlay.classList.add('opacity-100');
    document.getElementById('confirm-box').classList.remove('scale-95');
    document.getElementById('confirm-box').classList.add('scale-100');
}

function closeConfirm() {
    _pendingDeleteId = null;
    const overlay = document.getElementById('confirm-overlay');
    overlay.classList.add('opacity-0', 'pointer-events-none');
    overlay.classList.remove('opacity-100');
    document.getElementById('confirm-box').classList.add('scale-95');
    document.getElementById('confirm-box').classList.remove('scale-100');
}

async function executeConfirm() {
    if (!_pendingDeleteId) return;
    const sessionId = _pendingDeleteId;
    closeConfirm();
    try {
        if (sessionId === '__ALL__') {
            await fetch('/api/sessions', { method: 'DELETE' });
            activeSessionId = 'temp-' + Date.now();
            clearChatUI();
        } else {
            await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' });
            if (sessionId === activeSessionId) {
                activeSessionId = 'temp-' + Date.now();
                clearChatUI();
            }
        }
        await loadSessions();
        showToast('삭제되었습니다.');
    } catch (e) {
        console.error('Delete session failed:', e);
    }
}

function confirmDeleteAllSessions() {
    _pendingDeleteId = '__ALL__';
    document.getElementById('confirm-msg').textContent = '모든 세션을 삭제하시겠습니까?';
    const overlay = document.getElementById('confirm-overlay');
    overlay.classList.remove('opacity-0', 'pointer-events-none');
    overlay.classList.add('opacity-100');
    document.getElementById('confirm-box').classList.remove('scale-95');
    document.getElementById('confirm-box').classList.add('scale-100');
}

// ══════════════════════════════════════════════════════════════
// Sessions Panel
// ══════════════════════════════════════════════════════════════

function toggleSessionsPanel() {
    sessionsPanelOpen = !sessionsPanelOpen;
    const panel = document.getElementById('sessions-panel');
    const icon = document.getElementById('sessions-toggle-icon');
    if (sessionsPanelOpen) {
        panel.classList.remove('collapsed');
        icon.textContent = 'chevron_left';
        loadSessions();
    } else {
        panel.classList.add('collapsed');
        icon.textContent = 'chevron_right';
    }
}

function renderSessionPanel() {
    const container = document.getElementById('session-panel-list');
    // Only show sessions that have messages (non-empty)
    const nonEmpty = sessions.filter(s => !s.is_archived && s.messages && s.messages.length > 0);

    if (!nonEmpty.length) {
        container.innerHTML = '<p class="text-[15px] text-on-surface-variant p-4 text-center">대화 이력이 없습니다.</p>';
        return;
    }

    // Group by date
    const groups = {};
    nonEmpty.forEach(s => {
        const d = new Date(s.updated_at);
        const today = new Date();
        const yesterday = new Date(today);
        yesterday.setDate(yesterday.getDate() - 1);
        let key;
        if (d.toDateString() === today.toDateString()) key = 'Today';
        else if (d.toDateString() === yesterday.toDateString()) key = 'Yesterday';
        else key = `${d.getMonth() + 1}월 ${d.getDate()}일`;
        if (!groups[key]) groups[key] = [];
        groups[key].push(s);
    });

    container.innerHTML = Object.entries(groups).map(([label, items]) => `
        <div class="text-[11px] font-bold text-outline tracking-widest uppercase px-2 mb-2 mt-3">${label}</div>
        ${items.map(s => {
            const isActive = s.id === activeSessionId;
            const cls = isActive
                ? 'p-3 rounded-xl bg-surface-container-lowest shadow-sm border border-primary/10 cursor-pointer transition-all'
                : 'p-3 rounded-xl hover:bg-surface-container-high cursor-pointer transition-all group';
            const titleCls = isActive ? 'text-primary font-bold' : 'font-semibold group-hover:text-primary';
            return `
                <div class="${cls}" onclick="switchSession('${s.id}')">
                    <div class="flex justify-between items-center mb-1">
                        <span class="text-[10px] font-bold text-primary px-1.5 py-0.5 bg-primary-fixed rounded uppercase">Q&A</span>
                        <div class="flex items-center gap-1">
                            <span class="text-[11px] text-on-surface-variant">${s.relative_time}</span>
                            <button onclick="event.stopPropagation();confirmDeleteSession('${s.id}')" class="p-0.5 text-on-surface-variant hover:text-error rounded transition-colors opacity-0 group-hover:opacity-100 ${isActive ? 'opacity-100' : ''}">
                                <span class="material-symbols-outlined text-sm">close</span>
                            </button>
                        </div>
                    </div>
                    <div class="text-[15px] ${titleCls} line-clamp-1">${escapeHtml(s.title)}</div>
                    <div class="text-[13px] text-on-surface-variant line-clamp-1 mt-0.5">${escapeHtml(s.preview)}</div>
                </div>
            `;
        }).join('')}
    `).join('');
}

function filterSessions() {
    const q = document.getElementById('session-search').value.toLowerCase();
    const container = document.getElementById('session-panel-list');
    const filtered = sessions.filter(s =>
        !s.is_archived && s.messages && s.messages.length > 0 &&
        (s.title.toLowerCase().includes(q) || s.preview.toLowerCase().includes(q))
    );
    if (!filtered.length) {
        container.innerHTML = '<p class="text-[15px] text-on-surface-variant p-4 text-center">검색 결과가 없습니다.</p>';
        return;
    }
    container.innerHTML = filtered.map(s => {
        const isActive = s.id === activeSessionId;
        const cls = isActive
            ? 'p-3 rounded-xl bg-surface-container-lowest shadow-sm border border-primary/10 cursor-pointer transition-all'
            : 'p-3 rounded-xl hover:bg-surface-container-high cursor-pointer transition-all group';
        const titleCls = isActive ? 'text-primary font-bold' : 'font-semibold group-hover:text-primary';
        return `
            <div class="${cls}" onclick="switchSession('${s.id}')">
                <div class="flex justify-between items-center mb-1">
                    <span class="text-[10px] font-bold text-primary px-1.5 py-0.5 bg-primary-fixed rounded uppercase">Q&A</span>
                    <div class="flex items-center gap-1">
                        <span class="text-[11px] text-on-surface-variant">${s.relative_time}</span>
                        <button onclick="event.stopPropagation();confirmDeleteSession('${s.id}')" class="p-0.5 text-on-surface-variant hover:text-error rounded transition-colors opacity-0 group-hover:opacity-100 ${isActive ? 'opacity-100' : ''}">
                            <span class="material-symbols-outlined text-sm">close</span>
                        </button>
                    </div>
                </div>
                <div class="text-[15px] ${titleCls} line-clamp-1">${escapeHtml(s.title)}</div>
                <div class="text-[13px] text-on-surface-variant line-clamp-1 mt-0.5">${escapeHtml(s.preview)}</div>
            </div>
        `;
    }).join('');
}

// ══════════════════════════════════════════════════════════════
// Chat
// ══════════════════════════════════════════════════════════════

function handleInputKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
    updateSendButton();
}

function updateSendButton() {
    const input = document.getElementById('chat-input');
    const btn = document.getElementById('send-btn');
    btn.disabled = !input.value.trim() || isSending;
}

function askExample(el) {
    const icon = el.querySelector('.material-symbols-outlined');
    const text = el.textContent.replace(icon ? icon.textContent : '', '').trim();
    document.getElementById('chat-input').value = text;
    updateSendButton();
    sendMessage();
}

async function sendMessage() {
    const input = document.getElementById('chat-input');
    const question = input.value.trim();
    if (!question || isSending) return;

    isSending = true;
    input.value = '';
    updateSendButton();
    document.getElementById('empty-state').classList.add('hidden');
    appendUserMessage(question);
    const typingId = showTypingIndicator();
    const slowNoticeTimer = setTimeout(() => {
        updateTypingIndicator(
            typingId,
            '로컬 27B 모델이 CPU-only로 답변을 생성 중입니다. 첫 응답은 1~2분 이상 걸릴 수 있습니다.'
        );
        loadStatus();
    }, 8000);

    try {
        // Create real session if needed
        if (activeSessionId.startsWith('temp-')) {
            const sessRes = await fetch('/api/sessions', { method: 'POST' });
            const sess = await sessRes.json();
            activeSessionId = sess.id;
        }

        const res = await fetch('/api/chat/jobs', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: activeSessionId,
                question: question,
                top_k: settings.topK,
                relevance_threshold: settings.threshold,
                rerank_top_k: settings.rerankTopK,
                max_context: settings.maxContext,
            }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
            throw new Error(data.detail || `서버 오류 (${res.status})`);
        }
        activeJobId = data.job_id;
        updateTypingIndicator(typingId, jobProgressMessage(data));
        const completed = await pollChatJob(data.job_id, typingId);
        removeTypingIndicator(typingId);
        appendAIMessage(completed.answer || '', completed.evidences || [], completed.reference_materials || {}, completed.llm_sec || 0);
        await loadSessions();
    } catch (e) {
        removeTypingIndicator(typingId);
        appendAIMessage(`오류가 발생했습니다. ${escapeHtml(e.message || '서버 상태를 확인하세요.')}`, [], {}, 0);
        console.error('Chat error:', e);
    } finally {
        activeJobId = null;
        clearTimeout(slowNoticeTimer);
        await loadStatus();
        isSending = false;
        updateSendButton();
    }
}

async function pollChatJob(jobId, typingId) {
    while (true) {
        await sleep(1200);
        const res = await fetch(`/api/chat/jobs/${jobId}`);
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
            throw new Error(data.detail || `작업 상태 조회 실패 (${res.status})`);
        }
        updateTypingIndicator(typingId, jobProgressMessage(data));
        await loadStatus();
        if (data.status === 'done') return data;
        if (data.status === 'cancelled') throw new Error('답변 생성이 취소되었습니다.');
        if (data.status === 'error') throw new Error(data.error || '답변 생성 중 오류가 발생했습니다.');
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function jobProgressMessage(job) {
    const labels = {
        queued: '질문을 접수했습니다. 로컬 모델 실행 순서를 기다리는 중입니다.',
        retrieving: 'S1000D 문서를 검색 중입니다.',
        reranking: '후보 문서를 재정렬 중입니다.',
        generating: '로컬 LLM이 답변을 생성 중입니다. CPU-only 27B에서는 1~2분 이상 걸릴 수 있습니다.',
        done: '답변 생성이 완료되었습니다.',
        cancelled: '답변 생성이 취소되었습니다.',
        error: '답변 생성 중 오류가 발생했습니다.',
    };
    return labels[job.progress] || labels[job.status] || '작업을 처리 중입니다.';
}

async function cancelActiveJob() {
    if (!activeJobId) return;
    try {
        await fetch(`/api/chat/jobs/${activeJobId}`, { method: 'DELETE' });
    } catch (e) {
        console.error('Cancel job failed:', e);
    }
}

function clearChat() {
    if (activeSessionId && !activeSessionId.startsWith('temp-')) {
        confirmDeleteSession(activeSessionId);
    } else {
        createSession();
    }
}

function clearChatUI() {
    const canvas = document.getElementById('chat-canvas');
    // Keep only the empty state
    const emptyState = document.getElementById('empty-state');
    canvas.innerHTML = '';
    canvas.appendChild(emptyState);
    emptyState.classList.remove('hidden');
}

// ── Message rendering ──

function appendUserMessage(text) {
    const canvas = document.getElementById('chat-canvas');
    const div = document.createElement('div');
    div.className = 'flex flex-row-reverse items-start gap-4 msg-enter';
    div.innerHTML = `
        <div class="w-9 h-9 rounded-full bg-primary-container flex items-center justify-center text-white text-sm font-bold shrink-0">U</div>
        <div class="bg-primary text-white p-5 rounded-2xl rounded-tr-none max-w-xl shadow-lg shadow-primary/10">
            <p class="leading-relaxed text-base">${escapeHtml(text)}</p>
        </div>
    `;
    canvas.appendChild(div);
    scrollToBottom();
}


function renderReferenceMaterials(referenceMaterials) {
    if (!referenceMaterials || typeof referenceMaterials !== 'object') return '';

    const sections = [
        ['warnings', '안전주의', 'warning'],
        ['cautions', '주의사항', 'report'],
        ['references', '참조 문서', 'article'],
        ['procedures', '관련 절차', 'list_alt'],
        ['faults', '관련 고장', 'build_circle'],
        ['figures', '관련 그림', 'image'],
        ['graphic_assets', '그래픽 자산', 'perm_media'],
        ['hotspots', '핫스팟', 'touch_app'],
    ];

    const nonEmptySections = sections
        .map(([key, label, icon]) => ({ key, label, icon, items: referenceMaterials[key] || [] }))
        .filter(section => section.items.length > 0);

    if (nonEmptySections.length === 0) return '';

    const total = nonEmptySections.reduce((sum, section) => sum + section.items.length, 0);
    const body = nonEmptySections.map(section => `
        <div class="space-y-2">
            <div class="flex items-center gap-1.5 text-[13px] font-bold text-on-surface">
                <span class="material-symbols-outlined text-sm">${section.icon}</span>
                ${section.label} (${section.items.length}건)
            </div>
            <div class="space-y-1.5">
                ${section.items.map(item => renderReferenceMaterialItem(item)).join('')}
            </div>
        </div>
    `).join('<hr class="border-outline-variant/10 my-3"/>');

    const refId = 'ref-' + Date.now() + Math.random().toString(36).slice(2, 6);
    return `
        <div class="mt-4 bg-surface-container-lowest border border-outline-variant/10 rounded-xl overflow-hidden">
            <button onclick="toggleEvidence('${refId}')" class="w-full px-4 py-2.5 text-[13px] font-semibold text-on-surface-variant cursor-pointer hover:bg-surface-container-low flex items-center gap-2">
                <span class="material-symbols-outlined text-sm">fact_check</span>
                참고자료 (${total}건)
                <span class="material-symbols-outlined text-sm ml-auto evidence-chevron" id="${refId}-chevron">expand_more</span>
            </button>
            <div class="evidence-body" id="${refId}">
                <div class="px-4 py-3 border-t border-outline-variant/10">${body}</div>
            </div>
        </div>
    `;
}

function renderReferenceMaterialItem(item) {
    const title = item.title || item.label || item.dmc || item.id || 'reference';
    const meta = [item.dmc, item.relation, item.type, item.asset_format].filter(Boolean).join(' · ');
    const graphicPreview = renderGraphicAssetPreview(item, title);
    return `
        <div class="rounded-lg bg-surface-container-low px-3 py-2 text-[13px]">
            ${graphicPreview}
            <div class="font-semibold text-on-surface">${escapeHtml(title)}</div>
            ${meta ? `<div class="font-mono text-[11px] text-outline mt-0.5">${escapeHtml(meta)}</div>` : ''}
            ${item.text ? `<div class="text-on-surface-variant mt-1 line-clamp-2">${escapeHtml(item.text)}</div>` : ''}
        </div>
    `;
}

function renderGraphicAssetPreview(item, title) {
    if (!item || item.type !== 'GraphicAsset') return '';
    if (item.preview_url) {
        return `
            <a href="${escapeHtml(item.original_url || item.preview_url)}" target="_blank" rel="noopener" class="block mb-2 rounded-lg overflow-hidden border border-outline-variant/20 bg-white">
                <img src="${escapeHtml(item.preview_url)}" alt="${escapeHtml(title)}" class="w-full max-h-48 object-contain bg-white" loading="lazy"/>
            </a>
        `;
    }
    const statusLabels = {
        unsupported_cgm: 'CGM 형식은 브라우저 미리보기를 지원하지 않습니다.',
        missing: '미리보기 없음: 자산 파일을 찾을 수 없습니다.',
    };
    const label = statusLabels[item.preview_status] || '미리보기 없음';
    const download = item.original_url
        ? `<a href="${escapeHtml(item.original_url)}" target="_blank" rel="noopener" class="underline underline-offset-2">원본 열기</a>`
        : '';
    return `
        <div class="mb-2 rounded-lg border border-dashed border-outline-variant/30 bg-surface-container-lowest px-3 py-2 text-[12px] text-on-surface-variant">
            <span class="font-semibold">${escapeHtml(label)}</span>
            ${download}
        </div>
    `;
}

function formatEvidenceStrengthBadge(score) {
    const normalizedScore = Number.isFinite(Number(score)) ? Number(score) : 0;
    if (normalizedScore >= 0.6) {
        return {
            label: '근거 강함',
            className: 'text-success bg-success/10',
            title: '내부 검색 점수가 높은 참고 문서입니다. 답변 근거와 얼마나 잘 맞는지를 나타내는 내부 매칭 강도입니다.',
        };
    }
    if (normalizedScore >= 0.45) {
        return {
            label: '근거 적합',
            className: 'text-amber-700 bg-amber-100',
            title: '내부 검색 점수 기준으로 관련성이 있는 참고 문서입니다. 답변 근거와 얼마나 잘 맞는지를 나타내는 내부 매칭 강도입니다.',
        };
    }
    return {
        label: '참고 후보',
        className: 'text-outline bg-surface-container',
        title: '내부 검색 점수가 낮은 보조 참고 문서입니다. 답변 근거는 본문과 함께 확인하세요.',
    };
}

function appendAIMessage(text, evidences, referenceMaterials, llmSec) {
    const canvas = document.getElementById('chat-canvas');
    const div = document.createElement('div');
    div.className = 'flex items-start gap-4 msg-enter';

    let evidenceHtml = '';
    if (evidences && evidences.length > 0) {
        const cards = evidences.map(ev => {
            const strength = formatEvidenceStrengthBadge(ev.score);
            return `
                <div class="flex items-center gap-2 flex-wrap text-[13px]">
                    <span class="bg-primary text-white px-2 py-0.5 rounded-full font-bold">#${ev.rank}</span>
                    <span class="font-mono font-semibold text-on-surface">${escapeHtml(ev.dmc)}</span>
                    <span class="${strength.className} px-2 py-0.5 rounded-full font-bold" title="${escapeHtml(strength.title)}">${escapeHtml(strength.label)}</span>
                    ${ev.dm_type ? `<span class="bg-surface-container px-1.5 py-0.5 rounded text-on-surface-variant">${ev.dm_type}</span>` : ''}
                </div>
                ${ev.text ? `<p class="text-[13px] text-on-surface-variant line-clamp-2 mt-1">${escapeHtml(ev.text)}</p>` : ''}
            `;
        }).join('<hr class="border-outline-variant/10 my-2"/>');

        const evId = 'ev-' + Date.now() + Math.random().toString(36).slice(2, 6);
        evidenceHtml = `
            <div class="mt-4 bg-surface-container-lowest border border-outline-variant/10 rounded-xl overflow-hidden">
                <button onclick="toggleEvidence('${evId}')" class="w-full px-4 py-2.5 text-[13px] font-semibold text-on-surface-variant cursor-pointer hover:bg-surface-container-low flex items-center gap-2">
                    <span class="material-symbols-outlined text-sm">description</span>
                    참고 문서 (${evidences.length}건)
                    <span class="material-symbols-outlined text-sm ml-auto evidence-chevron" id="${evId}-chevron">expand_more</span>
                </button>
                <div class="evidence-body" id="${evId}">
                    <div>
                        <div class="px-4 py-3 space-y-2 border-t border-outline-variant/10">${cards}</div>
                    </div>
                </div>
            </div>
        `;
    }

    const referenceMaterialsHtml = renderReferenceMaterials(referenceMaterials);

    const metricsHtml = llmSec > 0
        ? `<div class="flex items-center gap-3 mt-3 text-[12px] text-outline">
               <span class="material-symbols-outlined text-xs">psychology</span> 추론 ${llmSec.toFixed(1)}s
           </div>`
        : '';

    // Convert markdown-like formatting
    const formattedText = formatAnswer(text);

    div.innerHTML = `
        <div class="w-9 h-9 rounded-xl bg-primary flex items-center justify-center shrink-0 shadow-lg shadow-primary/20">
            <span class="material-symbols-outlined text-white text-lg" style="font-variation-settings:'FILL' 1;">smart_toy</span>
        </div>
        <div class="flex-1 max-w-3xl">
            <div class="bg-surface-container-lowest p-6 rounded-2xl shadow-[0_10px_30px_rgba(25,28,30,0.04)] border border-outline-variant/10">
                <div class="prose text-on-surface leading-relaxed text-base">${formattedText}</div>
                <div class="flex gap-2 mt-4">
                    <button onclick="copyText(this)" class="px-3 py-1.5 bg-surface-container-high rounded-lg text-[13px] font-semibold hover:bg-surface-container-highest transition-colors flex items-center gap-1.5">
                        <span class="material-symbols-outlined text-xs">content_copy</span> Copy
                    </button>
                </div>
            </div>
            ${evidenceHtml}
            ${referenceMaterialsHtml}
            ${metricsHtml}
        </div>
    `;
    canvas.appendChild(div);
    scrollToBottom();
}

function showTypingIndicator() {
    const canvas = document.getElementById('chat-canvas');
    const id = 'typing-' + Date.now();
    const div = document.createElement('div');
    div.id = id;
    div.className = 'flex items-start gap-4 msg-enter';
    div.innerHTML = `
        <div class="w-9 h-9 rounded-xl bg-primary flex items-center justify-center shrink-0 shadow-lg shadow-primary/20">
            <span class="material-symbols-outlined text-white text-lg" style="font-variation-settings:'FILL' 1;">smart_toy</span>
        </div>
        <div class="p-4 bg-surface-container-lowest rounded-2xl border border-outline-variant/10">
            <div class="typing-dots flex gap-1.5">
                <span class="w-2 h-2 rounded-full bg-primary-container"></span>
                <span class="w-2 h-2 rounded-full bg-primary-container"></span>
                <span class="w-2 h-2 rounded-full bg-primary-container"></span>
            </div>
            <p class="typing-note hidden mt-3 text-[13px] text-on-surface-variant max-w-md"></p>
            <button onclick="cancelActiveJob()" class="mt-3 px-3 py-1.5 bg-surface-container-high rounded-lg text-[12px] font-semibold hover:bg-surface-container-highest transition-colors">
                취소
            </button>
        </div>
    `;
    canvas.appendChild(div);
    scrollToBottom();
    return id;
}

function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function updateTypingIndicator(id, message) {
    const el = document.getElementById(id);
    if (!el) return;
    const note = el.querySelector('.typing-note');
    if (!note) return;
    note.textContent = message;
    note.classList.remove('hidden');
}

function scrollToBottom() {
    const canvas = document.getElementById('chat-canvas');
    requestAnimationFrame(() => canvas.scrollTop = canvas.scrollHeight);
}

function toggleEvidence(id) {
    document.getElementById(id).classList.toggle('open');
    document.getElementById(id + '-chevron').classList.toggle('open');
}

function copyText(btn) {
    const text = btn.closest('.bg-surface-container-lowest').querySelector('.prose').textContent;
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text);
    } else {
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
    }
    btn.innerHTML = '<span class="material-symbols-outlined text-xs">check</span> Copied';
    setTimeout(() => {
        btn.innerHTML = '<span class="material-symbols-outlined text-xs">content_copy</span> Copy';
    }, 2000);
}

// ── Simple markdown formatting ──

function formatAnswer(text) {
    // Bold
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Headers
    text = text.replace(/^### (.*$)/gm, '<h4 class="font-bold text-on-surface mt-4 mb-2">$1</h4>');
    text = text.replace(/^## (.*$)/gm, '<h3 class="text-lg font-bold text-on-surface mt-4 mb-2">$1</h3>');
    // Lists
    text = text.replace(/^- (.*$)/gm, '<li class="ml-4 list-disc">$1</li>');
    text = text.replace(/^(\d+)\. (.*$)/gm, '<li class="ml-4 list-decimal">$2</li>');
    // Paragraphs
    text = text.replace(/\n\n/g, '</p><p class="mb-3">');
    text = text.replace(/\n/g, '<br/>');
    return `<p class="mb-3">${text}</p>`;
}

// ══════════════════════════════════════════════════════════════
// Settings
// ══════════════════════════════════════════════════════════════

function updateSettingLabel(key, value) {
    document.getElementById(`val-${key}`).textContent = value;
    // Update settings object
    const mapping = { topk: 'topK', threshold: 'threshold', rerank: 'rerankTopK', maxctx: 'maxContext' };
    if (mapping[key]) settings[mapping[key]] = parseFloat(value);
}

async function loadConfig() {
    try {
        const res = await fetch('/api/config');
        const cfg = await res.json();
        document.getElementById('cfg-temperature').textContent = cfg.temperature.toFixed(2);
        document.getElementById('cfg-top-p').textContent = cfg.top_p.toFixed(2);
        document.getElementById('cfg-repeat-penalty').textContent = cfg.repeat_penalty.toFixed(2);
        document.getElementById('cfg-max-tokens').textContent = cfg.max_tokens.toLocaleString();
        document.getElementById('cfg-n-ctx').textContent = cfg.n_ctx.toLocaleString();

        // Sync search sliders with server values
        settings.topK = cfg.top_k;
        settings.threshold = cfg.relevance_threshold;
        settings.rerankTopK = cfg.rerank_top_k;
        settings.maxContext = cfg.max_context_chars;
        document.getElementById('setting-topk').value = cfg.top_k;
        document.getElementById('val-topk').textContent = cfg.top_k;
        document.getElementById('setting-threshold').value = Math.round(cfg.relevance_threshold * 100);
        document.getElementById('val-threshold').textContent = cfg.relevance_threshold.toFixed(2);
        document.getElementById('setting-rerank').value = cfg.rerank_top_k;
        document.getElementById('val-rerank').textContent = cfg.rerank_top_k;
        document.getElementById('setting-maxctx').value = cfg.max_context_chars;
        document.getElementById('val-maxctx').textContent = cfg.max_context_chars;
    } catch (e) {
        console.error('Config load failed:', e);
    }
}

// ══════════════════════════════════════════════════════════════
// Utilities
// ══════════════════════════════════════════════════════════════

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showToast(msg) {
    const toast = document.getElementById('toast');
    document.getElementById('toast-msg').textContent = msg;
    toast.classList.remove('opacity-0', 'translate-y-4', 'pointer-events-none');
    toast.classList.add('opacity-100', 'translate-y-0');
    setTimeout(() => {
        toast.classList.add('opacity-0', 'translate-y-4', 'pointer-events-none');
        toast.classList.remove('opacity-100', 'translate-y-0');
    }, 2500);
}

// Auto-resize textarea
document.addEventListener('DOMContentLoaded', () => {
    const textarea = document.getElementById('chat-input');
    if (textarea) {
        textarea.addEventListener('input', () => {
            textarea.style.height = 'auto';
            textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
            updateSendButton();
        });
    }
});
