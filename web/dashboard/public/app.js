/**
 * Crypto Signal Bot Dashboard - Front-end JS
 * Supports: live updates via WebSocket + REST polling
 * Features: tabs (recommendations, all analyses, positions, closed trades),
 *          search/filter, modal for detailed analysis, P&L display
 */

const API = '/api';
const WS_URL = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;

// DOM
const $recs = document.getElementById('recommendations');
const $allAnalyses = document.getElementById('allAnalyses');
const $positions = document.getElementById('positionsList');
const $closedTrades = document.getElementById('closedTrades');
const $lastUpdate = document.getElementById('lastUpdate');
const $totalAnalyzed = document.getElementById('totalAnalyzed');
const $totalRecs = document.getElementById('totalRecs');
const $openPositions = document.getElementById('openPositions');
const $analysisTime = document.getElementById('analysisTime');
const $totalPnl = document.getElementById('totalPnl');
const $wsStatus = document.getElementById('wsStatus');
const $wsStatusText = document.getElementById('wsStatusText');
const $runNowBtn = document.getElementById('runNowBtn');
const $resetBtn = document.getElementById('resetBtn');
const $resetModal = document.getElementById('resetModal');
const $resetClose = document.getElementById('resetClose');
const $confirmResetBtn = document.getElementById('confirmResetBtn');
const $cancelResetBtn = document.getElementById('cancelResetBtn');
const $scanBottomsBtn = document.getElementById('scanBottomsBtn');
const $bottomCandidates = document.getElementById('bottomCandidates');
const $bottomScanInfo = document.getElementById('bottomScanInfo');
const $toast = document.getElementById('toast');
const $modal = document.getElementById('analysisModal');
const $modalBody = document.getElementById('modalBody');
const $modalClose = document.getElementById('modalClose');
const $search = document.getElementById('symbolSearch');
const $filterDirection = document.getElementById('filterDirection');
const $filterStatus = document.getElementById('filterStatus');
const $filterInfo = document.getElementById('filterInfo');

// State
let lastAnalysesData = null;

// ============================================================
// Helpers
// ============================================================
function fmtPrice(p) {
  if (p === null || p === undefined || isNaN(p)) return '-';
  if (p >= 1000) return p.toLocaleString('en-US', { maximumFractionDigits: 2 });
  if (p >= 1) return p.toFixed(4);
  if (p >= 0.01) return p.toFixed(6);
  return p.toFixed(10);
}

function fmtPct(p, withSign = true) {
  if (p === null || p === undefined || isNaN(p)) return '-';
  const sign = withSign && p > 0 ? '+' : '';
  return `${sign}${p.toFixed(2)}%`;
}

function escapeHtml(s) {
  if (!s) return '';
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[c]);
}

function showToast(message, type = 'info', duration = 4000) {
  $toast.textContent = message;
  $toast.className = `toast ${type} show`;
  setTimeout(() => $toast.classList.remove('show'), duration);
}

// ============================================================
// Tabs
// ============================================================
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(`tab-${tab.dataset.tab}`).classList.add('active');
  });
});

// ============================================================
// Modal
// ============================================================
$modalClose.addEventListener('click', () => $modal.classList.remove('show'));
$modal.addEventListener('click', (e) => {
  if (e.target === $modal) $modal.classList.remove('show');
});

function showAnalysisModal(symbol) {
  if (!lastAnalysesData) return;
  const a = lastAnalysesData.analyses.find(x => x.symbol === symbol);
  if (!a) return;

  const dirBadge = a.direction === 'bullish' ? '🟢 صاعد' :
                   a.direction === 'bearish' ? '🔴 هابط' : '⚪ محايد';

  let html = `
    <div class="modal-title">${a.symbol} <span class="badge ${a.direction}">${dirBadge}</span></div>

    <div class="modal-section">
      <div class="modal-section-title">📊 ملخص التحليل</div>
      <div class="modal-detail-grid">
        <div class="modal-detail">💰 السعر: <b>${fmtPrice(a.current_price)}</b></div>
        <div class="modal-detail">🎯 الثقة: <b>${a.confidence.toFixed(1)}%</b></div>
        <div class="modal-detail">📈 النتيجة: <b>${a.weighted_score.toFixed(1)}</b></div>
        <div class="modal-detail">🚀 الصعود المتوقع: <b>${fmtPct(a.expected_rise_pct)}</b></div>
        <div class="modal-detail">🛑 وقف الخسارة: <b>${fmtPrice(a.stop_loss)}</b></div>
        <div class="modal-detail">✅ جني الأرباح: <b>${fmtPrice(a.take_profit)}</b></div>
        <div class="modal-detail">⚖️ R/R: <b>${a.risk_reward_ratio.toFixed(2)}:1</b></div>
        <div class="modal-detail">📊 ATR: <b>${a.atr_pct.toFixed(2)}%</b></div>
        <div class="modal-detail">⏱️ الفريم: <b>${a.timeframe || '—'}</b></div>
        <div class="modal-detail">📅 التحليل: <b>${a.analyzed_at ? new Date(a.analyzed_at).toLocaleString('ar') : '—'}</b></div>
      </div>
    </div>
  `;

  if (a.is_recommended) {
    html += `
      <div class="modal-section">
        <div class="modal-section-title" style="color: var(--green);">✅ تم قبول التوصية</div>
        <div class="modal-detail" style="background: rgba(63, 185, 80, 0.1);">
          هذه التوصية ضمن قائمة التوصيات المقبولة. سيتم استخدامها لفتح صفقة ورقية عند توفر فرصة.
        </div>
      </div>
    `;
  } else {
    html += `
      <div class="modal-section">
        <div class="modal-section-title" style="color: var(--red);">❌ أسباب الرفض</div>
        ${a.rejection_reasons.map(r => `<div class="modal-detail" style="background: rgba(248, 81, 73, 0.1);">• ${escapeHtml(r)}</div>`).join('')}
      </div>
    `;
  }

  html += `
    <div class="modal-section">
      <div class="modal-section-title">🔬 تفصيل الاستراتيجيات (${a.strategy_breakdown.length})</div>
      ${a.strategy_breakdown.map(s => `
        <div class="modal-strategy ${s.direction}">
          <div class="modal-strategy-header">
            <span class="modal-strategy-name">${escapeHtml(s.strategy)}</span>
            <span class="modal-strategy-score">${s.direction} (${s.score.toFixed(1)}) — ${(s.confidence * 100).toFixed(0)}%</span>
          </div>
          <div class="modal-strategy-reasons">
            ${s.reasons.map(r => `<div>• ${escapeHtml(r)}</div>`).join('')}
          </div>
        </div>
      `).join('')}
    </div>
  `;

  $modalBody.innerHTML = html;
  $modal.classList.add('show');
}

// ============================================================
// Render functions
// ============================================================
function renderRecommendations(recs) {
  if (!recs || !recs.length) {
    $recs.innerHTML = '<div class="empty">لا توجد توصيات قوية حالياً. اضغط "تشغيل التحليل الآن" أو انتظر الدورة القادمة.</div>';
    return;
  }
  $recs.innerHTML = recs.map(r => {
    const dir = r.direction || 'neutral';
    const dirText = dir === 'bullish' ? 'صاعد 🟢' : dir === 'bearish' ? 'هابط 🔴' : 'محايد ⚪';
    const conf = r.confidence || 0;
    const reasons = (r.signals || [])
      .flatMap(s => s.reasons || [])
      .slice(0, 6)
      .map(rsn => `<div class="reason">${escapeHtml(rsn)}</div>`)
      .join('');
    return `
      <div class="card ${dir}">
        <div class="card-header">
          <span class="symbol">${r.symbol}</span>
          <span class="badge ${dir}">${dirText}</span>
        </div>
        <div class="card-body">
          <div class="price-row">
            <span class="label">💰 السعر الحالي</span>
            <span class="value">${fmtPrice(r.current_price)}</span>
          </div>
          <div class="price-row">
            <span class="label">📈 الصعود المتوقع</span>
            <span class="value green">${fmtPct(r.expected_rise_pct)}</span>
          </div>
          <div class="price-row">
            <span class="label">🎯 الثقة</span>
            <span class="value">${conf.toFixed(1)}%</span>
          </div>
          <div class="confidence-bar">
            <div class="confidence-fill" style="width:${conf}%"></div>
          </div>
          <div class="price-row">
            <span class="label">🛑 وقف الخسارة</span>
            <span class="value red">${fmtPrice(r.stop_loss)}</span>
          </div>
          <div class="price-row">
            <span class="label">✅ جني الأرباح</span>
            <span class="value green">${fmtPrice(r.take_profit)}</span>
          </div>
          <div class="price-row">
            <span class="label">⚖️ R/R</span>
            <span class="value">${(r.risk_reward_ratio || 0).toFixed(2)}:1</span>
          </div>
          <div class="reasons">
            <div class="reasons-title">أبرز الإشارات</div>
            ${reasons || '<div class="reason">لا توجد إشارات محددة</div>'}
          </div>
        </div>
      </div>
    `;
  }).join('');
}

function renderAllAnalyses(data) {
  lastAnalysesData = data;
  if (!data || !data.analyses || !data.analyses.length) {
    $allAnalyses.innerHTML = '<div class="empty">لا توجد بيانات تحليل بعد. شغّل دورة تحليل أولاً.</div>';
    $filterInfo.textContent = '—';
    return;
  }
  // Apply filters
  const search = $search.value.toUpperCase();
  const dirFilter = $filterDirection.value;
  const statusFilter = $filterStatus.value;
  let filtered = data.analyses;
  if (search) filtered = filtered.filter(a => a.symbol.includes(search));
  if (dirFilter) filtered = filtered.filter(a => a.direction === dirFilter);
  if (statusFilter === 'recommended') filtered = filtered.filter(a => a.is_recommended);
  if (statusFilter === 'rejected') filtered = filtered.filter(a => !a.is_recommended);

  $filterInfo.textContent = `${filtered.length} / ${data.analyses.length} عملة`;

  if (!filtered.length) {
    $allAnalyses.innerHTML = '<div class="empty">لا توجد نتائج تطابق الفلاتر.</div>';
    return;
  }

  $allAnalyses.innerHTML = filtered.map(a => {
    const statusClass = a.is_recommended ? 'recommended' : 'rejected';
    const statusIcon = a.is_recommended ? '✅' : '❌';
    const dirClass = a.direction;
    const dirText = a.direction === 'bullish' ? 'صاعد' :
                    a.direction === 'bearish' ? 'هابط' : 'محايد';
    const reasons = a.is_recommended ?
      (a.strategy_breakdown || []).flatMap(s => s.reasons).slice(0, 2) :
      (a.rejection_reasons || []).slice(0, 2);
    return `
      <div class="analysis-row ${statusClass}" data-symbol="${a.symbol}">
        <span class="a-symbol">${a.symbol}</span>
        <span class="a-direction ${dirClass}">${dirText}</span>
        <span class="a-confidence">${a.confidence.toFixed(0)}%</span>
        <span class="a-score">${a.weighted_score.toFixed(0)}</span>
        <span class="a-reasons">${reasons.map(r => escapeHtml(r)).join(' • ')}</span>
        <span class="a-status ${statusClass}" title="${a.is_recommended ? 'مقبول' : 'مرفوض'}">${statusIcon}</span>
      </div>
    `;
  }).join('');

  // Attach click handlers
  $allAnalyses.querySelectorAll('.analysis-row').forEach(row => {
    row.addEventListener('click', () => showAnalysisModal(row.dataset.symbol));
  });
}

function renderPositions(positions) {
  if (!positions || !positions.length) {
    $positions.innerHTML = '<div class="empty glass">لا توجد صفقات مفتوحة. عند فتح البوت لصفقات ورقية، ستظهر هنا مع تتبع P&L حي وشريط تقدم السعر.</div>';
    $openPositions.textContent = '0';
    $totalPnl.textContent = '$0.00';
    return;
  }
  $openPositions.textContent = positions.length;
  const totalPnl = positions.reduce((sum, p) => sum + (p.current_pnl || 0), 0);
  $totalPnl.textContent = `$${totalPnl.toFixed(2)}`;
  $totalPnl.style.color = totalPnl >= 0 ? 'var(--green)' : 'var(--red)';

  $positions.innerHTML = positions.map(p => {
    const pnlClass = p.current_pnl >= 0 ? 'green' : 'red';
    const modeTag = p.paper ? 'ورقي' : '🔴 حقيقي';
    const updatesTag = p.updates_count > 0 ? ` · ${p.updates_count} تحديث` : '';
    const pnlText = p.current_price ?
      `${fmtPct(p.current_pnl_pct)} ($${p.current_pnl.toFixed(4)})` : '—';
    const totalFees = p.total_fees ? `$${p.total_fees.toFixed(4)}` : '$0';

    // PRICE PROGRESS BAR (NEW!)
    let progressBar = '';
    if (p.current_price && p.stop_loss && p.take_profit && p.direction === 'bullish') {
      const progress = Math.max(0, Math.min(100, p.progress_pct || 0));
      const progressColor = progress >= 50 ? 'var(--green)' : progress >= 30 ? 'var(--yellow)' : 'var(--red)';
      progressBar = `
        <div class="price-progress">
          <div class="price-progress-header">
            <span>تقدم السعر نحو الهدف</span>
            <span style="color: ${progressColor}; font-weight: 700;">${progress.toFixed(1)}%</span>
          </div>
          <div class="price-progress-track">
            <div class="price-progress-marker" style="left: ${progress}%;"></div>
          </div>
          <div class="price-progress-labels">
            <span class="sl">🛑 SL ${fmtPrice(p.stop_loss)}</span>
            <span class="entry">📍Entry ${fmtPrice(p.entry_price)}</span>
            <span class="tp">✅ TP ${fmtPrice(p.take_profit)}</span>
          </div>
        </div>
      `;
    }

    return `
      <div class="position-card">
        <div class="position-header">
          <span class="p-symbol">${p.symbol} <small style="color: var(--text-muted)">${modeTag}${updatesTag}</small></span>
          <span class="value ${pnlClass}">${pnlText}</span>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; font-size: 12px; margin-bottom: 8px;">
          <div><span class="label">💰 السعر الحالي</span><br><span class="value">${p.current_price ? fmtPrice(p.current_price) : '—'}</span></div>
          <div><span class="label">📍 الدخول</span><br><span class="value">${fmtPrice(p.entry_price)}</span></div>
          <div><span class="label">📏 الحجم</span><br><span class="value">${(p.size || 0).toFixed(6)} ($${(p.notional_usd || 0).toFixed(2)})</span></div>
          <div><span class="label">💸 الرسوم</span><br><span class="value" style="color: var(--red);">${totalFees}</span></div>
        </div>
        ${progressBar}
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; font-size: 11px; margin-top: 8px; color: var(--text-muted);">
          <div>SL: <span style="color: var(--red);">${fmtPrice(p.stop_loss)} (${p.sl_distance_pct?.toFixed(2)}%)</span></div>
          <div>TP: <span style="color: var(--green);">${fmtPrice(p.take_profit)} (${p.tp_distance_pct?.toFixed(2)}%)</span></div>
        </div>
      </div>
    `;
  }).join('');
}

function renderClosedTrades(trades) {
  if (!trades || !trades.length) {
    $closedTrades.innerHTML = '<div class="empty">لا توجد صفقات مغلقة بعد.</div>';
    return;
  }
  // Show most recent first
  const sorted = [...trades].reverse();
  $closedTrades.innerHTML = sorted.map(t => {
    const win = (t.pnl || 0) >= 0;
    const pnlClass = win ? 'win' : 'loss';
    const exitTime = t.exit_time ? new Date(t.exit_time).toLocaleString('ar') : '—';
    return `
      <div class="closed-row ${win ? 'win' : 'loss'}">
        <span class="p-symbol">${t.symbol}</span>
        <span>${t.paper ? 'ورقي' : '🔴'}</span>
        <span>دخول: ${fmtPrice(t.entry_price)}</span>
        <span>خروج: ${fmtPrice(t.exit_price)}</span>
        <span class="closed-pnl ${pnlClass}">${fmtPct(t.pnl_pct)}<br><small>$${(t.pnl || 0).toFixed(2)}</small></span>
        <span>${exitTime}</span>
        <span style="color: var(--text-muted)">${escapeHtml(t.reason || '')}</span>
      </div>
    `;
  }).join('');
}

// ============================================================
// Bottom candidates
// ============================================================
function renderBottomCandidates(data) {
  if (!data || !data.top_candidates || !data.top_candidates.length) {
    $bottomCandidates.innerHTML = '<div class="empty">لا توجد عملات في القاع بعد. اضغط "فحص القاع الآن" للبحث.</div>';
    $bottomScanInfo.textContent = data?.timestamp ? `آخر فحص: ${new Date(data.timestamp).toLocaleString('ar')}` : '—';
    return;
  }
  $bottomScanInfo.textContent = `${data.top_candidates.length} عملة · ${data.timestamp ? new Date(data.timestamp).toLocaleString('ar') : ''}`;
  $bottomCandidates.innerHTML = data.top_candidates.map((c, i) => {
    const score = c.score || 0;
    const scoreColor = score >= 70 ? 'var(--green)' : score >= 50 ? 'var(--yellow)' : 'var(--text-muted)';
    const distColor = c.distance_from_low_pct < 2 ? 'var(--red)' : 'var(--text-muted)';
    const rsiColor = c.rsi < 30 ? 'var(--green)' : c.rsi < 40 ? 'var(--yellow)' : 'var(--text-muted)';
    const reasons = (c.signals || []).slice(0, 5).map(r => `<div class="reason">${escapeHtml(r)}</div>`).join('');
    const patterns = (c.patterns_detected || []).slice(0, 3).map(p => `<span class="badge bullish" style="margin-left:4px;font-size:10px;">${escapeHtml(p)}</span>`).join('');
    return `
      <div class="card bullish" style="border-right: 3px solid ${scoreColor};">
        <div class="card-header">
          <span class="symbol">#${i+1} · ${c.symbol}</span>
          <span class="badge bullish" style="background: ${scoreColor}20; color: ${scoreColor};">Score: ${score.toFixed(0)}</span>
        </div>
        <div class="card-body">
          <div class="price-row">
            <span class="label">💰 السعر الحالي</span>
            <span class="value">${fmtPrice(c.current_price)}</span>
          </div>
          <div class="price-row">
            <span class="label">📉 القاع الأخير</span>
            <span class="value" style="color: ${distColor};">${fmtPrice(c.recent_low)} (${c.distance_from_low_pct.toFixed(1)}% فوقه)</span>
          </div>
          <div class="price-row">
            <span class="label">📊 RSI</span>
            <span class="value" style="color: ${rsiColor};">${c.rsi?.toFixed(1) || '—'}</span>
          </div>
          <div class="price-row">
            <span class="label">📐 BB %B</span>
            <span class="value">${c.bb_percent_b?.toFixed(2) || '—'}</span>
          </div>
          <div class="price-row">
            <span class="label">🛑 وقف الخسارة</span>
            <span class="value red">${fmtPrice(c.stop_loss)}</span>
          </div>
          <div class="price-row">
            <span class="label">✅ جني الأرباح</span>
            <span class="value green">${fmtPrice(c.take_profit)}</span>
          </div>
          <div class="price-row">
            <span class="label">⚖️ R/R</span>
            <span class="value">${(c.risk_reward_ratio || 0).toFixed(2)}:1</span>
          </div>
          ${patterns ? `<div class="price-row"><span class="label">🕯️ أنماط</span><span>${patterns}</span></div>` : ''}
          <div class="reasons">
            <div class="reasons-title">إشارات الارتداد (${c.signals?.length || 0})</div>
            ${reasons || '<div class="reason">لا توجد إشارات</div>'}
          </div>
        </div>
      </div>
    `;
  }).join('');
}

async function fetchBottomCandidates() {
  try {
    const r = await fetch(`${API}/bottom-candidates`);
    const data = await r.json();
    renderBottomCandidates(data);
  } catch (e) {
    console.error('Bottom fetch error:', e);
  }
}

// ============================================================
// Statistics (database-backed)
// ============================================================
async function fetchStats() {
  try {
    const [summary, strategies, runs] = await Promise.all([
      fetch(`${API}/stats/summary`).then(r => r.json()),
      fetch(`${API}/stats/strategies?days=30`).then(r => r.json()),
      fetch(`${API}/stats/runs?limit=10`).then(r => r.json()),
    ]);
    renderStatsSummary(summary);
    renderStrategyStats(strategies.strategies || []);
    renderRunsHistory(runs.runs || []);
  } catch (e) {
    console.error('Stats fetch error:', e);
  }
}

function renderStatsSummary(s) {
  if (!s || s.error) return;
  document.getElementById('statTotalRuns').textContent = s.total_runs || 0;
  document.getElementById('statTotalRecs').textContent = s.total_recommendations || 0;
  document.getElementById('statClosed').textContent = s.total_closed_trades || 0;
  const winRate = s.win_rate || 0;
  const $winRate = document.getElementById('statWinRate');
  $winRate.textContent = `${winRate.toFixed(1)}%`;
  $winRate.style.color = winRate >= 50 ? 'var(--green)' : 'var(--red)';
  const totalPnl = s.total_pnl || 0;
  const $totalPnl = document.getElementById('statTotalPnl');
  $totalPnl.textContent = `$${totalPnl.toFixed(2)}`;
  $totalPnl.style.color = totalPnl >= 0 ? 'var(--green)' : 'var(--red)';
  const $avgPnl = document.getElementById('statAvgPnl');
  $avgPnl.textContent = `${(s.avg_pnl_pct || 0).toFixed(2)}%`;
  $avgPnl.style.color = (s.avg_pnl_pct || 0) >= 0 ? 'var(--green)' : 'var(--red)';
}

function renderStrategyStats(strategies) {
  const $stats = document.getElementById('strategyStats');
  if (!strategies.length) {
    $stats.innerHTML = '<div class="empty">لا توجد إشارات استراتيجية بعد. اضغط زر التحليل لجمع البيانات.</div>';
    return;
  }
  // Sort by total signals desc
  strategies.sort((a, b) => (b.total_signals || 0) - (a.total_signals || 0));
  $stats.innerHTML = strategies.map(s => {
    const bullPct = s.total_signals ? (s.bullish_signals / s.total_signals * 100).toFixed(1) : 0;
    return `
      <div class="analysis-row recommended">
        <span class="a-symbol">${escapeHtml(s.strategy_name || '')}</span>
        <span class="a-direction bullish">${s.bullish_signals || 0}🟢 / ${s.bearish_signals || 0}🔴</span>
        <span class="a-confidence">${s.total_signals || 0}</span>
        <span class="a-score">${(s.avg_score || 0).toFixed(1)}</span>
        <span class="a-reasons">Bullish ${bullPct}% · Avg conf ${(s.avg_confidence * 100 || 0).toFixed(0)}%</span>
        <span class="a-status recommended" title="Signals count">${s.total_signals || 0}</span>
      </div>
    `;
  }).join('');
}

function renderRunsHistory(runs) {
  const $runs = document.getElementById('runsHistory');
  if (!runs.length) {
    $runs.innerHTML = '<div class="empty">لا توجد دورات بعد.</div>';
    return;
  }
  $runs.innerHTML = runs.map(r => {
    const time = r.timestamp ? new Date(r.timestamp).toLocaleString('ar') : '—';
    return `
      <div class="analysis-row">
        <span class="a-symbol">${time}</span>
        <span class="a-direction neutral">${r.mode || 'paper'}</span>
        <span class="a-confidence">${r.signals_passed || 0}</span>
        <span class="a-score">${r.recommendations_count || 0}</span>
        <span class="a-reasons">${r.symbols_analyzed || 0} عملة · ${r.duration_seconds || 0}s · ${r.bottom_candidates_count || 0} قاع</span>
        <span class="a-status" style="color: var(--text-muted);">${r.recommendations_count || 0}</span>
      </div>
    `;
  }).join('');
}

function updateStats(data) {
  const recs = (data && data.top_recommendations) || [];
  $totalRecs.textContent = recs.length;
  $totalAnalyzed.textContent = (data && data.symbols_analyzed) || 0;
  $analysisTime.textContent = (data && data.analysis_time_seconds) ? `${data.analysis_time_seconds}s` : '—';
  $lastUpdate.textContent = (data && data.timestamp) ? new Date(data.timestamp).toLocaleString('ar') : '—';
}

// ============================================================
// Data fetching
// ============================================================
async function fetchRecommendations() {
  try {
    const r = await fetch(`${API}/recommendations`);
    const data = await r.json();
    renderRecommendations(data.top_recommendations || []);
    updateStats(data);
  } catch (e) {
    console.error('Recs fetch error:', e);
  }
}

async function fetchAllAnalyses() {
  try {
    const r = await fetch(`${API}/all-analyses`);
    const data = await r.json();
    renderAllAnalyses(data);
  } catch (e) {
    console.error('All-analyses fetch error:', e);
  }
}

async function fetchPositions() {
  try {
    const r = await fetch(`${API}/positions`);
    const data = await r.json();
    renderPositions(data);
  } catch (e) {
    console.error('Positions fetch error:', e);
  }
}

async function fetchClosedTrades() {
  try {
    const r = await fetch(`${API}/closed-trades`);
    const data = await r.json();
    renderClosedTrades(data);
  } catch (e) {
    console.error('Closed fetch error:', e);
  }
}

// v5.2: leader/follower market map (BTC/ETH/SOL/XRP groups)
const TREND_BADGE = {
  bullish: ['🟢 صاعد', 'var(--green)'],
  bearish: ['🔴 هابط', 'var(--red)'],
  neutral: ['⚪ محايد', 'var(--text-muted)'],
};

async function fetchMarketMap() {
  try {
    const r = await fetch(`${API}/market-map`);
    renderMarketMap(await r.json());
  } catch (e) {
    console.error('Market map fetch error:', e);
  }
}

// v5.4: human-readable classification file viewer (decision-making aid)
const $groupsFileBtn = document.getElementById('groupsFileBtn');
if ($groupsFileBtn) {
  $groupsFileBtn.addEventListener('click', async () => {
    const $box = document.getElementById('groupsFileBox');
    if ($box.style.display === 'none' || !$box.style.display) {
      if (!$box.dataset.loaded) {
        try {
          const r = await fetch(`${API}/market-groups-file`);
          $box.textContent = await r.text();
          $box.dataset.loaded = '1';
        } catch (e) {
          $box.textContent = 'تعذر جلب الملف: ' + e.message;
        }
      }
      $box.style.display = 'block';
      $groupsFileBtn.textContent = '📄 إخفاء ملف التصنيف';
    } else {
      $box.style.display = 'none';
      $groupsFileBtn.textContent = '📄 عرض ملف التصنيف (market_groups.txt)';
    }
  });
}

// v5.4: market cycle verdict (leader-coin deep read) above the groups
const CYCLE_BADGE = {
  strong_bullish: ['🟢🟢 صاعد بقوة', 'var(--green)'],
  bullish: ['🟢 صاعد', 'var(--green)'],
  neutral: ['⚪ متوازن', 'var(--text-muted)'],
  bearish: ['🔴 هابط', 'var(--red)'],
  strong_bearish: ['🔴🔴 هابط بقوة', 'var(--red)'],
};

function renderMarketCycle(cycle) {
  const $panel = document.getElementById('marketCyclePanel');
  if (!$panel) return;
  if (!cycle || !cycle.market) { $panel.style.display = 'none'; return; }
  const mk = cycle.market;
  const [vLabel, vColor] = CYCLE_BADGE[mk.verdict] || CYCLE_BADGE.neutral;
  document.getElementById('mcVerdict').innerHTML =
    `<span style="color:${vColor}">${vLabel} (${mk.score}/100)</span>`;
  document.getElementById('mcPosture').textContent = mk.posture_ar || '';
  const leaders = cycle.leaders || {};
  document.getElementById('mcLeaders').innerHTML = Object.entries(leaders).map(([ld, a]) => {
    const [ll, lc] = CYCLE_BADGE[a.verdict] || CYCLE_BADGE.neutral;
    const chg = a.chg_24h_pct != null ? `${a.chg_24h_pct > 0 ? '+' : ''}${a.chg_24h_pct}%` : '—';
    return `<span class="glass" style="padding:6px 10px; display:inline-block;">` +
      `<b>${escapeHtml(ld.replace('USDT', ''))}</b> ` +
      `<span style="color:${lc}; font-weight:600;">${ll}</span>` +
      `<span style="color:var(--text-muted); font-size:0.85em;"> · RSI ${a.rsi14} · 24h ${chg} · ${a.followers} تابع</span></span>`;
  }).join('');
  document.getElementById('mcPolicy').innerHTML =
    (mk.policy || []).map(p => `<li>${escapeHtml(p.text_ar || '')}</li>`).join('');
  $panel.style.display = 'block';
}

function renderMarketMap(data) {
  const $map = document.getElementById('marketMap');
  renderMarketCycle(data && data.cycle);
  if (!data || data.error || !data.groups) {
    $map.innerHTML = '<div class="empty glass">لا توجد بيانات تصنيف بعد.</div>';
    return;
  }
  const leaders = data.leaders || {};
  const groups = data.groups || {};
  const order = Object.keys(leaders).filter(ld => groups[ld]);
  if (groups['independent']) order.push('independent');
  Object.keys(groups).filter(g => !order.includes(g)).forEach(g => order.push(g));
  if (!order.length) {
    $map.innerHTML = '<div class="empty glass">لا توجد بيانات تصنيف بعد.</div>';
    return;
  }
  const upd = data.updated_at ? new Date(data.updated_at).toLocaleString('ar') : '';
  const hours = data.refresh_hours || 6;
  document.getElementById('marketMapUpdated').textContent =
    upd ? `آخر تحديث: ${upd} · تُحدّث كل ${hours} ساعات` : '';

  $map.innerHTML = order.map(g => {
    const isLeader = g !== 'independent';
    const meta = leaders[g] || {};
    const [label, color] = isLeader
      ? (TREND_BADGE[meta.trend] || TREND_BADGE.neutral)
      : ['دون قائد', 'var(--text-muted)'];
    const followers = groups[g] || [];
    const shown = followers.slice(0, 12);
    const rows = shown.map(f => `
      <div class="analysis-row">
        <span class="a-symbol">${escapeHtml(f.symbol)}</span>
        <span class="a-direction neutral">corr ${f.corr != null ? (f.corr * 100).toFixed(0) : '—'}%</span>
        <span class="a-score">beta ${f.beta != null ? Number(f.beta).toFixed(2) : '—'}</span>
      </div>`).join('');
    const more = followers.length > shown.length
      ? `<div class="empty" style="padding:6px;">+${followers.length - shown.length} عملة أخرى</div>` : '';
    const chg = isLeader && meta.chg_24h_pct != null
      ? ` · 24h ${meta.chg_24h_pct > 0 ? '+' : ''}${meta.chg_24h_pct}%` : '';
    return `
      <div class="glass" style="margin-bottom:12px; padding:12px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
          <b>${escapeHtml(g === 'independent' ? 'عملات مستقلة' : g)}</b>
          <span style="color:${color}; font-weight:600;">${label}${chg}</span>
        </div>
        ${isLeader ? `<div style="color:var(--text-muted); font-size:0.85em; margin-bottom:6px;">${followers.length} عملة تتبع ${escapeHtml(g)} — إذا هبط ${escapeHtml(g)} فغالباً يهبطون معه</div>` : ''}
        ${rows}${more}
      </div>`;
  }).join('');
}

async function fetchAll() {
  await Promise.all([
    fetchRecommendations(),
    fetchAllAnalyses(),
    fetchPositions(),
    fetchClosedTrades(),
    fetchBottomCandidates(),
    fetchStats(),
    fetchMarketMap(),
  ]);
}

// ============================================================
// Run analysis button
// ============================================================
$runNowBtn.addEventListener('click', async () => {
  if ($runNowBtn.disabled) return;
  $runNowBtn.disabled = true;
  $runNowBtn.classList.add('running');
  $runNowBtn.textContent = '⏳ جاري التحليل...';
  showToast('بدأ التحليل. قد يستغرق 1-2 دقيقة.', 'info');
  try {
    const res = await fetch('/api/run-analysis', { method: 'POST' });
    const data = await res.json();
    if (data.status === 'started') {
      showToast('✅ بدأت دورة التحليل. تحديث تلقائي خلال 30 ثانية.', 'success', 5000);
      // Poll all data every 15s for up to 4 minutes
      let attempts = 0;
      const maxAttempts = 16;
      const interval = setInterval(async () => {
        attempts++;
        await fetchAll();
        if (attempts >= maxAttempts) {
          clearInterval(interval);
          showToast('انتهى التحديث التلقائي. اضغط الزر لإعادة التحليل.', 'info');
        }
      }, 15000);
    } else {
      showToast('استجابة غير متوقعة: ' + JSON.stringify(data), 'error');
    }
  } catch (e) {
    showToast('فشل الطلب: ' + e.message, 'error');
  } finally {
    setTimeout(() => {
      $runNowBtn.disabled = false;
      $runNowBtn.classList.remove('running');
      $runNowBtn.textContent = '▶ تشغيل التحليل الآن';
    }, 60000);
  }
});

// ============================================================
// Search/filter listeners
// ============================================================
$search.addEventListener('input', () => { if (lastAnalysesData) renderAllAnalyses(lastAnalysesData); });
$filterDirection.addEventListener('change', () => { if (lastAnalysesData) renderAllAnalyses(lastAnalysesData); });
$filterStatus.addEventListener('change', () => { if (lastAnalysesData) renderAllAnalyses(lastAnalysesData); });

// ============================================================
// Scan bottoms button
// ============================================================
$scanBottomsBtn?.addEventListener('click', async () => {
  if ($scanBottomsBtn.disabled) return;
  $scanBottomsBtn.disabled = true;
  $scanBottomsBtn.classList.add('running');
  $scanBottomsBtn.textContent = '⏳ جاري الفحص...';
  showToast('بدأ فحص القاع. قد يستغرق 1-2 دقيقة.', 'info');
  try {
    const res = await fetch('/api/scan-bottoms', { method: 'POST' });
    const data = await res.json();
    if (data.status === 'started') {
      showToast('✅ بدأ فحص القاع. تحديث تلقائي خلال 30 ثانية.', 'success', 5000);
      // Poll every 15s for 4 minutes
      let attempts = 0;
      const maxAttempts = 16;
      const interval = setInterval(async () => {
        attempts++;
        await fetchBottomCandidates();
        if (attempts >= maxAttempts) {
          clearInterval(interval);
        }
      }, 15000);
    }
  } catch (e) {
    showToast('فشل الطلب: ' + e.message, 'error');
  } finally {
    setTimeout(() => {
      $scanBottomsBtn.disabled = false;
      $scanBottomsBtn.classList.remove('running');
      $scanBottomsBtn.textContent = '🔍 فحص القاع الآن';
    }, 60000);
  }
});

// ============================================================
// Reset history button
// ============================================================
$resetBtn.addEventListener('click', () => $resetModal.classList.add('show'));
$resetClose.addEventListener('click', () => $resetModal.classList.remove('show'));
$cancelResetBtn.addEventListener('click', () => $resetModal.classList.remove('show'));
$resetModal.addEventListener('click', (e) => {
  if (e.target === $resetModal) $resetModal.classList.remove('show');
});

$confirmResetBtn.addEventListener('click', async () => {
  $confirmResetBtn.disabled = true;
  $confirmResetBtn.innerHTML = '⏳ جاري التصفير...';
  showToast('بدأ تصفير المحفوظات...', 'info');
  try {
    const res = await fetch('/api/reset-history', { method: 'POST' });
    const data = await res.json();
    if (data.status === 'success') {
      showToast('✅ تم تصفير كل المحفوظات بنجاح!', 'success', 5000);
      $resetModal.classList.remove('show');
      // Refresh all data
      await fetchAll();
    } else {
      showToast('خطأ: ' + JSON.stringify(data), 'error', 8000);
    }
  } catch (e) {
    showToast('فشل الطلب: ' + e.message, 'error');
  } finally {
    $confirmResetBtn.disabled = false;
    $confirmResetBtn.innerHTML = '🗑️ نعم، صفّر كل شيء';
  }
});

// ============================================================
// Lazy loading: only fetch data for active tab
// ============================================================
let activeTab = 'recommendations';
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    tab.classList.add('active');
    const tabId = tab.dataset.tab;
    document.getElementById(`tab-${tabId}`).classList.add('active');
    activeTab = tabId;
    // Fetch data for this tab only
    fetchTabData(tabId);
  });
});

async function fetchTabData(tabId) {
  switch (tabId) {
    case 'recommendations':
    case 'all-analyses':
      await Promise.all([fetchRecommendations(), fetchAllAnalyses()]);
      break;
    case 'bottoms':
      await fetchBottomCandidates();
      break;
    case 'positions':
      await fetchPositions();
      break;
    case 'closed':
      await fetchClosedTrades();
      break;
    case 'stats':
      await fetchStats();
      break;
  }
}

// ============================================================
// WebSocket
// ============================================================
function connectWS() {
  let ws;
  let reconnectInterval;
  function connect() {
    ws = new WebSocket(WS_URL);
    ws.onopen = () => {
      $wsStatus.classList.add('connected');
      $wsStatus.classList.remove('error');
      $wsStatusText.textContent = 'متصل';
      if (reconnectInterval) clearInterval(reconnectInterval);
    };
    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.type === 'recommendations' && msg.data) {
          renderRecommendations(msg.data.top_recommendations || []);
          updateStats(msg.data);
          // Lazy: only refresh active tab
          if (activeTab === 'all-analyses') fetchAllAnalyses();
          if (activeTab === 'positions') fetchPositions();
        }
      } catch (err) {
        console.error('WS message error:', err);
      }
    };
    ws.onerror = () => {
      $wsStatus.classList.add('error');
      $wsStatus.classList.remove('connected');
      $wsStatusText.textContent = 'خطأ';
    };
    ws.onclose = () => {
      $wsStatus.classList.remove('connected');
      $wsStatusText.textContent = 'إعادة الاتصال…';
      reconnectInterval = setInterval(connect, 5000); // longer retry (5s)
    };
  }
  connect();
}

// ============================================================
// Init
// ============================================================
async function init() {
  // Initial load: only fetch recommendations + stats (active tab)
  await Promise.all([fetchRecommendations(), fetchPositions()]);
  connectWS();
  // Reduced polling: positions every 60s (was 30s)
  setInterval(() => { if (activeTab === 'positions') fetchPositions(); }, 60000);
  // Full refresh every 2 min as fallback (was 60s)
  setInterval(() => fetchTabData(activeTab), 120000);
}

// Start
init();
