/**
 * Mechano-Velocity — Interactive Dashboard Client
 * 
 * Handles: view routing, API calls, pipeline status polling, result rendering.
 * Supports both live pipeline runs and pre-computed Colab results.
 */

// ============================================================
// Plot Metadata — maps filenames to titles and categories
// ============================================================
const PLOT_META = {
    // Preprocessing
    'tissue_image':               { title: 'Tissue H&E Image',              category: 'preprocessing', wide: true },
    'qc_histograms':              { title: 'QC Histograms',                 category: 'preprocessing' },
    'spatial_qc':                 { title: 'Spatial QC Metrics',            category: 'preprocessing' },
    'umap_clusters':              { title: 'UMAP Clusters',                 category: 'preprocessing' },
    'key_genes_spatial':          { title: 'Key Genes (Spatial)',            category: 'preprocessing', wide: true },
    // Mechanotyping
    'gene_expression_spatial':    { title: 'Gene Expression (Spatial)',      category: 'mechanotyping', wide: true },
    'resistance_map':             { title: 'ECM Resistance Field',          category: 'mechanotyping' },
    'resistance_distribution':    { title: 'Resistance Distribution',       category: 'mechanotyping' },
    'resistance_histology_overlay':{ title: 'Resistance vs Histology',      category: 'mechanotyping' },
    'resistance_categories':      { title: 'Resistance Categories',         category: 'mechanotyping', wide: true },
    'drug_simulation':            { title: 'Drug Simulation (LOX)',         category: 'mechanotyping', wide: true },
    'cluster_resistance':         { title: 'Cluster Resistance Ranking',    category: 'mechanotyping' },
    // Graph & Velocity
    'spatial_graph':              { title: 'Spatial Graph',                 category: 'velocity' },
    'velocity_arrows':            { title: 'Velocity Arrows',              category: 'velocity' },
    'velocity_streamplot':        { title: 'Velocity Streamlines',         category: 'velocity' },
    'analysis_overview':          { title: 'Analysis Overview',            category: 'velocity', wide: true },
    'flow_by_cluster':            { title: 'Flow Analysis by Cluster',     category: 'velocity', wide: true },
    'velocity_distribution':      { title: 'Velocity Distribution',        category: 'velocity' },
    // Clinical & Validation
    'clinical_overview':          { title: 'Clinical Overview',            category: 'clinical', wide: true },
    'tumor_classification':       { title: 'Tumor Classification',         category: 'clinical' },
    'validation_correlation':     { title: 'Resistance-Velocity Correlation', category: 'clinical' },
    'validation_wall_vs_fluid':   { title: 'Wall vs Fluid Velocity',       category: 'clinical' },
    'ablation_study':             { title: 'Ablation Study',               category: 'clinical', wide: true },
    'validation_overlay':         { title: 'Validation Overlay',           category: 'clinical' },
};

// ============================================================
// State
// ============================================================
const state = {
    currentView: 'upload',
    polling: null,
    datasetLoaded: false,
    uploadedFiles: [],
    currentPlotCategory: 'all',
    allPlots: {},
};

// ============================================================
// API Helpers
// ============================================================
const API = {
    async get(endpoint) {
        const res = await fetch(`/api${endpoint}`);
        return res.json();
    },

    async post(endpoint, data = {}) {
        const res = await fetch(`/api${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        return res.json();
    },

    async upload(endpoint, formData) {
        const res = await fetch(`/api${endpoint}`, {
            method: 'POST',
            body: formData,
        });
        return res.json();
    },
};

// ============================================================
// View Routing
// ============================================================
function switchView(viewName) {
    state.currentView = viewName;
    
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    
    const view = document.getElementById(`view-${viewName}`);
    if (view) view.classList.add('active');
    
    const link = document.querySelector(`[data-view="${viewName}"]`);
    if (link) link.classList.add('active');
    
    // Load data for the view
    if (viewName === 'results') loadResults();
    if (viewName === 'history') loadHistory();
}

// Nav link click handlers
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', () => switchView(link.dataset.view));
});

// ============================================================
// Upload View
// ============================================================
const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');
const filesList = document.getElementById('upload-files-list');
const btnUpload = document.getElementById('btn-upload');
const btnUseSample = document.getElementById('btn-use-sample');

// Drag & drop
uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    handleFiles(e.dataTransfer.files);
});

fileInput.addEventListener('change', (e) => {
    handleFiles(e.target.files);
});

function handleFiles(fileList) {
    state.uploadedFiles = Array.from(fileList);
    renderFileList();
    btnUpload.disabled = state.uploadedFiles.length === 0;
}

function renderFileList() {
    filesList.innerHTML = state.uploadedFiles
        .map(f => `<span class="file-tag">${f.name}</span>`)
        .join('');
}

// Upload button
btnUpload.addEventListener('click', async () => {
    btnUpload.disabled = true;
    btnUpload.textContent = 'Uploading...';
    
    const formData = new FormData();
    const datasetName = document.getElementById('dataset-name').value || 'uploaded_dataset';
    formData.append('dataset_name', datasetName);
    
    for (const file of state.uploadedFiles) {
        if (file.name.endsWith('.h5') || file.name.endsWith('.h5ad')) {
            formData.append('h5_file', file);
        } else {
            formData.append('spatial_files', file);
        }
    }
    
    try {
        const result = await API.upload('/upload', formData);
        if (result.success) {
            showDatasetStatus(datasetName, result);
        } else {
            alert('Upload failed: ' + result.error);
        }
    } catch (err) {
        alert('Upload error: ' + err.message);
    }
    
    btnUpload.disabled = false;
    btnUpload.textContent = 'Upload Files';
});

// Use sample dataset
btnUseSample.addEventListener('click', async () => {
    btnUseSample.disabled = true;
    btnUseSample.textContent = 'Loading...';
    
    try {
        const result = await API.post('/use-sample');
        if (result.success) {
            showDatasetStatus(result.dataset_name, result);
        } else {
            alert('Error: ' + result.error);
        }
    } catch (err) {
        alert('Error: ' + err.message);
    }
    
    btnUseSample.disabled = false;
    btnUseSample.textContent = 'Use Sample Data';
});

function showDatasetStatus(name, result) {
    state.datasetLoaded = true;
    const statusDiv = document.getElementById('dataset-status');
    statusDiv.style.display = 'block';
    document.getElementById('dataset-info').innerHTML = `
        <p><strong>Dataset:</strong> ${name}</p>
        <p><strong>Path:</strong> <code>${result.path}</code></p>
    `;
}

// Go to pipeline
document.getElementById('btn-go-pipeline').addEventListener('click', () => switchView('pipeline'));

// ============================================================
// Pipeline View
// ============================================================
const btnRunAll = document.getElementById('btn-run-all');
const progressFill = document.getElementById('pipeline-progress-fill');
const progressText = document.getElementById('pipeline-progress-text');

// Run individual stages
document.getElementById('btn-run-preprocess').addEventListener('click', () => runStage('preprocess'));
document.getElementById('btn-run-mechanotype').addEventListener('click', () => runStage('mechanotype'));
document.getElementById('btn-run-graph').addEventListener('click', () => runStage('graph'));
document.getElementById('btn-run-clinical').addEventListener('click', () => runStage('clinical'));

// Run full pipeline
btnRunAll.addEventListener('click', async () => {
    btnRunAll.disabled = true;
    clearLogs();
    
    try {
        await API.post('/run/full');
        startPolling();
    } catch (err) {
        alert('Error starting pipeline: ' + err.message);
        btnRunAll.disabled = false;
    }
});

async function runStage(stage) {
    clearLogs();
    try {
        await API.post(`/run/${stage}`);
        startPolling();
    } catch (err) {
        alert('Error: ' + err.message);
    }
}

// ============================================================
// Status Polling
// ============================================================
function startPolling() {
    if (state.polling) clearInterval(state.polling);
    state.polling = setInterval(pollStatus, 1500);
    pollStatus(); // immediate first call
}

function stopPolling() {
    if (state.polling) {
        clearInterval(state.polling);
        state.polling = null;
    }
}

async function pollStatus() {
    try {
        const data = await API.get('/status');
        updatePipelineUI(data);
    } catch (err) {
        // Server might be busy
    }
}

function updatePipelineUI(data) {
    const status = data.status || {};
    const progress = status.progress || 0;
    const message = status.message || '';
    const stages = status.stages_completed || [];
    const currentStage = status.current_stage;
    const error = status.error;
    const logs = status.logs || [];
    
    // Progress bar
    progressFill.style.width = `${progress}%`;
    progressText.textContent = `${progress}% — ${message}`;
    
    // Nav status indicator
    const dot = document.querySelector('.status-dot');
    const statusText = document.querySelector('.status-text');
    
    if (error) {
        dot.className = 'status-dot error';
        statusText.textContent = 'Error';
    } else if (progress >= 100) {
        dot.className = 'status-dot complete';
        statusText.textContent = 'Complete';
        stopPolling();
        btnRunAll.disabled = false;
    } else if (progress > 0) {
        dot.className = 'status-dot running';
        statusText.textContent = message.substring(0, 30);
    } else {
        dot.className = 'status-dot idle';
        statusText.textContent = 'Idle';
    }
    
    // Stage cards
    const allStages = ['preprocessing', 'mechanotyping', 'graph_velocity', 'clinical'];
    allStages.forEach(s => {
        const card = document.getElementById(`stage-${s}`);
        const badge = document.querySelector(`#status-${s} .stage-badge`);
        if (!card || !badge) return;
        
        card.classList.remove('active', 'complete', 'error');
        
        if (stages.includes(s)) {
            card.classList.add('complete');
            badge.className = 'stage-badge complete';
            badge.textContent = '✓ Complete';
        } else if (currentStage === s) {
            card.classList.add('active');
            badge.className = 'stage-badge running';
            badge.textContent = 'Running...';
        } else if (error && currentStage === s) {
            card.classList.add('error');
            badge.className = 'stage-badge error';
            badge.textContent = 'Error';
        } else {
            badge.className = 'stage-badge pending';
            badge.textContent = 'Pending';
        }
    });
    
    // Logs
    if (logs.length > 0) {
        renderLogs(logs);
    }
    
    // Auto-switch to results when done
    if (progress >= 100 && state.currentView === 'pipeline') {
        setTimeout(() => switchView('results'), 1500);
    }
}

function clearLogs() {
    const container = document.getElementById('log-container');
    container.innerHTML = '';
}

function renderLogs(logs) {
    const container = document.getElementById('log-container');
    container.innerHTML = logs.map(log => {
        let cls = 'log-entry';
        if (log.includes('ERROR')) cls += ' error';
        if (log.includes('Complete') || log.includes('Saved')) cls += ' success';
        return `<div class="${cls}">${escapeHtml(log)}</div>`;
    }).join('');
    container.scrollTop = container.scrollHeight;
}

// ============================================================
// Results View
// ============================================================
async function loadResults() {
    try {
        const data = await API.get('/results');
        
        // Render report
        if (data.report) {
            renderReport(data.report);
        }
        
        // Render validation
        if (data.report && data.report.validation) {
            renderValidation(data.report.validation);
        }
        
        // Render plots with categories
        if (data.plots && Object.keys(data.plots).length > 0) {
            state.allPlots = data.plots;
            document.getElementById('plot-tabs').style.display = 'flex';
            renderPlots(data.plots, 'all');
            setupPlotTabs();
        }
    } catch (err) {
        console.error('Error loading results:', err);
    }
}

function renderReport(report) {
    const card = document.getElementById('card-report');
    card.style.display = 'block';
    
    const scores = report.scores || {};
    const classification = report.classification || {};
    const cellCounts = report.cell_counts || {};
    
    // Badge
    const badge = document.getElementById('report-badge');
    const cat = (classification.risk_category || '').toLowerCase();
    if (cat.includes('hot')) {
        badge.className = 'report-badge hot';
        badge.textContent = '🔥 HOT';
    } else if (cat.includes('cold')) {
        badge.className = 'report-badge cold';
        badge.textContent = '❄️ COLD';
    } else {
        badge.className = 'report-badge intermediate';
        badge.textContent = '⚡ INTERMEDIATE';
    }
    
    // Scores
    document.getElementById('report-scores').innerHTML = `
        <div class="score-card">
            <div class="score-value">${(scores.mts || 0).toFixed(4)}</div>
            <div class="score-label">Mechano-Therapeutic Score</div>
        </div>
        <div class="score-card">
            <div class="score-value">${(scores.metastatic_risk || 0).toFixed(4)}</div>
            <div class="score-label">Metastatic Risk</div>
        </div>
        <div class="score-card">
            <div class="score-value">${(scores.immune_exclusion || 0).toFixed(4)}</div>
            <div class="score-label">Immune Exclusion</div>
        </div>
    `;
    
    // Recommendation
    document.getElementById('report-recommendation').innerHTML = `
        <strong>📋 ${classification.risk_category || 'N/A'}</strong><br/>
        ${classification.recommendation || ''}
    `;
    
    // Details
    document.getElementById('report-details').innerHTML = `
        Tumor spots: <strong>${cellCounts.tumor_spots || 0}</strong> · 
        T-cell spots: <strong>${cellCounts.tcell_spots || 0}</strong> · 
        Boundary spots: <strong>${cellCounts.boundary_spots || 0}</strong> · 
        Mean boundary resistance: <strong>${(report.mean_boundary_resistance || 0).toFixed(4)}</strong>
    `;
}

function renderValidation(validation) {
    const card = document.getElementById('card-validation');
    card.style.display = 'block';
    const grid = document.getElementById('validation-grid');
    
    let html = '';
    
    // Resistance-velocity correlation
    const rv = validation.resistance_velocity_correlation;
    if (rv) {
        html += `
            <div class="validation-card ${rv.result === 'PASS' ? 'pass' : 'fail'}">
                <h4>📊 Resistance-Velocity Correlation</h4>
                <div class="val-stat">Pearson r: <strong>${rv.pearson_r}</strong></div>
                <div class="val-stat">P-value: <strong>${rv.p_value.toExponential(2)}</strong></div>
                <div class="val-stat">${rv.interpretation}</div>
                <div class="val-result ${rv.result === 'PASS' ? 'pass' : 'fail'}">✅ ${rv.result}</div>
            </div>
        `;
    }
    
    // Wall vs fluid velocity
    const wf = validation.wall_vs_fluid_velocity;
    if (wf) {
        html += `
            <div class="validation-card ${wf.result === 'PASS' ? 'pass' : 'fail'}">
                <h4>📊 Wall vs Fluid Velocity</h4>
                <div class="val-stat">Wall mean: <strong>${wf.wall_mean_velocity.toFixed(4)}</strong></div>
                <div class="val-stat">Fluid mean: <strong>${wf.fluid_mean_velocity.toFixed(4)}</strong></div>
                <div class="val-stat">T-stat: <strong>${wf.t_statistic.toFixed(4)}</strong> · P: <strong>${wf.p_value.toExponential(2)}</strong></div>
                <div class="val-result ${wf.result === 'PASS' ? 'pass' : 'fail'}">✅ ${wf.result}</div>
            </div>
        `;
    }
    
    // Ablation study
    const ab = validation.ablation_study;
    if (ab) {
        html += `
            <div class="validation-card pass">
                <h4>📊 Ablation: Effect of Resistance Correction</h4>
                <div class="val-stat">Corrected mean velocity: <strong>${ab.corrected_mean_velocity.toFixed(4)}</strong></div>
                <div class="val-stat">Uncorrected mean velocity: <strong>${ab.uncorrected_mean_velocity.toFixed(4)}</strong></div>
                <div class="val-stat">${ab.interpretation}</div>
            </div>
        `;
    }
    
    grid.innerHTML = html;
}

function setupPlotTabs() {
    document.querySelectorAll('.plot-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.plot-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            state.currentPlotCategory = tab.dataset.category;
            renderPlots(state.allPlots, tab.dataset.category);
        });
    });
}

function renderPlots(plots, category) {
    const grid = document.getElementById('plots-grid');
    
    let html = '';
    for (const [key, url] of Object.entries(plots)) {
        const meta = PLOT_META[key] || { title: key.replace(/_/g, ' '), category: 'other' };
        
        // Filter by category
        if (category !== 'all' && meta.category !== category) continue;
        
        const wideClass = meta.wide ? ' wide' : '';
        html += `
            <div class="plot-card${wideClass}" data-category="${meta.category}">
                <img src="${url}" alt="${meta.title}" loading="lazy" />
                <div class="plot-title">${meta.title}</div>
            </div>
        `;
    }
    
    grid.innerHTML = html || '<p class="empty-state">No plots in this category.</p>';
}

// ============================================================
// Drug Simulation
// ============================================================
const drugSlider = document.getElementById('drug-reduction');
const drugVal = document.getElementById('drug-reduction-val');

drugSlider.addEventListener('input', () => {
    drugVal.textContent = `${drugSlider.value}%`;
});

document.getElementById('btn-drug-sim').addEventListener('click', async () => {
    const btn = document.getElementById('btn-drug-sim');
    btn.disabled = true;
    btn.textContent = 'Running...';
    
    const target = document.getElementById('drug-target').value;
    const reduction = parseInt(drugSlider.value);
    
    try {
        await API.post('/run/drug-sim', {
            target_gene: target,
            reduction_pct: reduction,
        });
        
        // Poll for completion
        let attempts = 0;
        const check = setInterval(async () => {
            attempts++;
            const status = await API.get('/status');
            
            if (!status.status.current_stage || attempts > 40) {
                clearInterval(check);
                btn.disabled = false;
                btn.textContent = 'Run Simulation';
                
                // Reload results to show new plot
                loadResults();
            }
        }, 2000);
    } catch (err) {
        alert('Drug simulation error: ' + err.message);
        btn.disabled = false;
        btn.textContent = 'Run Simulation';
    }
});

// ============================================================
// History View
// ============================================================
async function loadHistory() {
    try {
        const data = await API.get('/history');
        const runs = data.runs || [];
        const tbody = document.getElementById('history-tbody');
        
        if (runs.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="empty-state">No analysis runs yet.</td></tr>';
            return;
        }
        
        tbody.innerHTML = runs.map(run => {
            const report = run.clinical_report;
            return `
                <tr>
                    <td>#${run.id}</td>
                    <td>${run.sample_id || '—'}</td>
                    <td>${formatDate(run.run_timestamp)}</td>
                    <td><span class="stage-badge ${run.status === 'completed' ? 'complete' : 'pending'}">${run.status}</span></td>
                    <td>${report ? report.mts_score?.toFixed(2) : '—'}</td>
                    <td>${report ? report.risk_category : '—'}</td>
                </tr>
            `;
        }).join('');
    } catch (err) {
        console.error('Error loading history:', err);
    }
}

// ============================================================
// Utilities
// ============================================================
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatDate(iso) {
    if (!iso) return '—';
    try {
        const d = new Date(iso);
        return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch {
        return iso;
    }
}

// ============================================================
// Init
// ============================================================
async function init() {
    try {
        const status = await API.get('/status');
        
        if (status.checkpoints) {
            const hasAny = Object.values(status.checkpoints).some(v => v);
            if (hasAny) state.datasetLoaded = true;
        }
        
        // If there are already plots available, show a hint
        if (status.plots && Object.keys(status.plots).length > 0) {
            const dot = document.querySelector('.status-dot');
            const statusText = document.querySelector('.status-text');
            dot.className = 'status-dot complete';
            statusText.textContent = 'Results ready';
        }
        
        // If pipeline is running, start polling
        if (status.status && status.status.progress > 0 && status.status.progress < 100) {
            switchView('pipeline');
            startPolling();
        }
    } catch (err) {
        console.log('API server not reachable — start it with: python api_server.py');
    }
}

init();
