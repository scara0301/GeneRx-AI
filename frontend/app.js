/**
 * GeneRx AI — Frontend Application
 * Vanilla JS SPA with Doctor & Patient modes
 */

// Hugging Face unified deployment dynamic path detection
// HF Spaces load in an iframe at paths like /embed/userName/spaceName
const basePath = window.location.pathname.replace(/\/$/, "");
const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://localhost:8000'
    : basePath;
let selectedDrugs = new Set();
let drugCatalog = [];
let patientWizardStep = 1;
let patientData = {};

// ── Init ──────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    lucide.createIcons();
    checkApiStatus();
    loadDrugCatalog();
    setupEventListeners();
});

function setupEventListeners() {
    // Mode toggle
    document.getElementById('btnDoctor').addEventListener('click', () => switchMode('doctor'));
    document.getElementById('btnPatient').addEventListener('click', () => switchMode('patient'));

    // Run assessment
    document.getElementById('runAssessment').addEventListener('click', runAssessment);

    // Clear form
    document.getElementById('clearForm').addEventListener('click', resetForm);
}

// ── API ───────────────────────────────────────────────────────────────────

async function checkApiStatus() {
    const dot = document.getElementById('apiStatus');
    const text = document.getElementById('apiStatusText');
    try {
        const resp = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(5000) });
        if (resp.ok) {
            const data = await resp.json();
            dot.className = 'status-dot online';
            text.textContent = data.ml_model_loaded ? 'ML Model Active' : 'Rules Engine Only';
        } else throw new Error();
    } catch {
        dot.className = 'status-dot offline';
        text.textContent = 'API Offline';
    }
}

async function loadDrugCatalog() {
    try {
        const resp = await fetch(`${API_BASE}/api/drugs`);
        const data = await resp.json();
        drugCatalog = data.drugs;
        renderDrugGrid();
        renderCurrentMedsCheckboxes();
    } catch {
        // Fallback catalog if API is not running
        drugCatalog = [
            { name: 'Metformin', category: 'Antidiabetic' },
            { name: 'Atorvastatin', category: 'Statin' },
            { name: 'Amlodipine', category: 'Calcium Channel Blocker' },
            { name: 'Ramipril', category: 'ACE Inhibitor' },
            { name: 'Metoprolol', category: 'Beta Blocker' },
            { name: 'Warfarin', category: 'Anticoagulant' },
            { name: 'Amoxicillin', category: 'Antibiotic' },
            { name: 'Ibuprofen', category: 'NSAID' },
            { name: 'Paracetamol', category: 'Analgesic' },
            { name: 'Omeprazole', category: 'PPI' },
        ];
        renderDrugGrid();
        renderCurrentMedsCheckboxes();
    }
}

// ── Mode Switching ────────────────────────────────────────────────────────

function switchMode(mode) {
    currentMode = mode;
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(mode === 'doctor' ? 'btnDoctor' : 'btnPatient').classList.add('active');

    const sidebar = document.getElementById('sidebar');
    const drugCard = document.getElementById('drugSelectionCard');
    const resultsCard = document.getElementById('resultsCard');
    const wizard = document.getElementById('patientWizard');

    if (mode === 'doctor') {
        sidebar.classList.remove('hidden');
        drugCard.classList.remove('hidden');
        resultsCard.classList.add('hidden');
        wizard.classList.add('hidden');
    } else {
        sidebar.classList.add('hidden');
        drugCard.classList.add('hidden');
        resultsCard.classList.add('hidden');
        wizard.classList.remove('hidden');
        patientWizardStep = 1;
        renderPatientWizard();
    }
}

// ── Drug Grid ─────────────────────────────────────────────────────────────

function renderDrugGrid() {
    const grid = document.getElementById('drugGrid');
    grid.innerHTML = drugCatalog.map(d => `
        <div class="drug-card ${selectedDrugs.has(d.name) ? 'selected' : ''}"
             data-drug="${d.name}" onclick="toggleDrug('${d.name}')">
            <div class="drug-card-name">${d.name}</div>
            <div class="drug-card-cat">${d.category || d.description || ''}</div>
        </div>
    `).join('');
    updateRunButton();
}

function toggleDrug(name) {
    if (selectedDrugs.has(name)) {
        selectedDrugs.delete(name);
    } else {
        selectedDrugs.add(name);
    }
    renderDrugGrid();
}

function updateRunButton() {
    const btn = document.getElementById('runAssessment');
    btn.disabled = selectedDrugs.size === 0;
}

function renderCurrentMedsCheckboxes() {
    const grid = document.getElementById('currentMedsGrid');
    grid.innerHTML = drugCatalog.map(d => `
        <label class="check-item">
            <input type="checkbox" value="${d.name}">
            <span>${d.name}</span>
        </label>
    `).join('');
}

// ── Collect Form Data ─────────────────────────────────────────────────────

function collectPatientProfile() {
    const conditions = [];
    document.querySelectorAll('#conditionsGrid input:checked').forEach(
        cb => conditions.push(cb.value)
    );
    const currentMeds = [];
    document.querySelectorAll('#currentMedsGrid input:checked').forEach(
        cb => currentMeds.push(cb.value)
    );
    const allergies = [];
    document.querySelectorAll('#allergiesGrid input:checked').forEach(
        cb => allergies.push(cb.value)
    );

    return {
        name: document.getElementById('patientName').value || 'Patient',
        age: parseInt(document.getElementById('patientAge').value) || 50,
        sex: document.getElementById('patientSex').value,
        weight_kg: parseFloat(document.getElementById('patientWeight').value) || 70,
        conditions,
        egfr: parseFloat(document.getElementById('labEgfr').value) || 90,
        alt: parseFloat(document.getElementById('labAlt').value) || 25,
        ast: parseFloat(document.getElementById('labAlt').value) || 25,
        hba1c: parseFloat(document.getElementById('labHba1c').value) || 5.5,
        systolic_bp: parseInt(document.getElementById('labSbp').value) || 120,
        diastolic_bp: Math.round((parseInt(document.getElementById('labSbp').value) || 120) * 0.65),
        ldl: parseFloat(document.getElementById('labLdl').value) || 3.0,
        inr: parseFloat(document.getElementById('labInr').value) || 1.0,
        current_meds: currentMeds,
        allergies,
    };
}

// ── Run Assessment ────────────────────────────────────────────────────────

async function runAssessment() {
    const btn = document.getElementById('runAssessment');
    btn.classList.add('loading');
    btn.innerHTML = '<span>Analyzing...</span>';

    const patient = collectPatientProfile();
    const drugs = Array.from(selectedDrugs);

    try {
        const resp = await fetch(`${API_BASE}/api/assess`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ patient, drugs }),
        });

        if (!resp.ok) throw new Error(`API returned ${resp.status}`);
        const data = await resp.json();
        renderResults(data);
    } catch (err) {
        console.error('Assessment error:', err);
        // Show error in results
        const resultsCard = document.getElementById('resultsCard');
        resultsCard.classList.remove('hidden');
        resultsCard.innerHTML = `
            <div class="card-header">
                <h2>Connection Error</h2>
                <p class="card-subtitle">Could not reach the API server. Make sure the backend is running on port 8000.</p>
            </div>
        `;
    } finally {
        btn.classList.remove('loading');
        btn.innerHTML = '<i data-lucide="activity"></i><span>Run Assessment</span>';
        lucide.createIcons();
    }
}

// ── Render Results ────────────────────────────────────────────────────────

function renderResults(data) {
    const card = document.getElementById('resultsCard');
    card.classList.remove('hidden');
    card.classList.add('fade-in');

    const assessments = data.assessments || [];
    const interactions = data.interactions || [];

    // Meta
    document.getElementById('resultsMeta').innerHTML =
        `<span class="card-subtitle">${assessments.length} drug${assessments.length > 1 ? 's' : ''} assessed for ${data.patient_name || 'Patient'}</span>`;

    // Summary table
    renderSummaryTable(assessments);

    // DDI
    renderInteractions(interactions);

    // Detail cards
    renderDetailCards(assessments);

    // ML chart
    renderMLChart(assessments);

    // Scroll to results
    card.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function riskClass(suitability) {
    const map = {
        'Suitable': 'safe',
        'Caution': 'caution',
        'Avoid': 'avoid',
        'Contraindicated': 'critical',
    };
    return map[suitability] || 'caution';
}

function renderSummaryTable(assessments) {
    const wrap = document.getElementById('summaryTable');
    wrap.innerHTML = `
        <table class="results-table">
            <thead>
                <tr>
                    <th>Drug</th>
                    <th>Suitability</th>
                    <th>Risk Level</th>
                    <th>ML Confidence</th>
                    <th>Key Concern</th>
                </tr>
            </thead>
            <tbody>
                ${assessments.map(a => {
        const rc = riskClass(a.suitability);
        const mlConf = a.ml_prediction?.available
            ? `${a.ml_prediction.confidence}%`
            : 'N/A';
        const concern = a.reasons?.[0] || 'None identified';
        return `
                    <tr>
                        <td><strong>${a.drug_name}</strong></td>
                        <td><span class="risk-badge ${rc}"><span class="risk-dot ${rc}"></span>${a.suitability}</span></td>
                        <td>${a.risk_level || ''}</td>
                        <td>${mlConf}</td>
                        <td style="max-width:260px;font-size:0.8rem;color:var(--text-secondary)">${concern}</td>
                    </tr>`;
    }).join('')}
            </tbody>
        </table>
    `;
}

function renderInteractions(interactions) {
    const section = document.getElementById('ddiSection');
    const list = document.getElementById('ddiList');

    if (!interactions.length) {
        section.classList.add('hidden');
        return;
    }

    section.classList.remove('hidden');
    list.innerHTML = interactions.map(ix => `
        <div class="ddi-item">
            <div class="ddi-drugs">${ix.drug_a} + ${ix.drug_b} (${ix.severity})</div>
            <div>${ix.message}</div>
        </div>
    `).join('');
}

function renderDetailCards(assessments) {
    const container = document.getElementById('detailCards');
    container.innerHTML = assessments.map((a, i) => {
        const rc = riskClass(a.suitability);
        const seHtml = a.side_effects
            ? Object.keys(a.side_effects).map(se => `<span class="se-tag">${se.split('(')[0].trim()}</span>`).join('')
            : '';

        return `
        <div class="detail-card ${i === 0 ? 'expanded' : ''}" id="detail-${i}">
            <div class="detail-card-header" onclick="toggleDetail(${i})">
                <div class="detail-card-title">
                    <span class="risk-dot ${rc}"></span>
                    ${a.drug_name}
                </div>
                <div class="detail-chevron"><i data-lucide="chevron-down"></i></div>
            </div>
            <div class="detail-card-body">
                <div class="detail-row">
                    <div class="detail-label">Reasoning</div>
                    <div class="detail-value">
                        <ul class="detail-list">
                            ${(a.reasons || []).map(r => `<li>${r}</li>`).join('')}
                        </ul>
                    </div>
                </div>
                ${a.dose_notes?.length ? `
                <div class="detail-row">
                    <div class="detail-label">Dose Notes</div>
                    <div class="detail-value">
                        <ul class="detail-list">
                            ${a.dose_notes.map(d => `<li>${d}</li>`).join('')}
                        </ul>
                    </div>
                </div>` : ''}
                ${a.warnings?.length ? `
                <div class="detail-row">
                    <div class="detail-label">Warnings</div>
                    <div class="detail-value" style="color:var(--avoid)">
                        <ul class="detail-list">
                            ${a.warnings.map(w => `<li>${w}</li>`).join('')}
                        </ul>
                    </div>
                </div>` : ''}
                ${seHtml ? `
                <div class="detail-row">
                    <div class="detail-label">Side Effects</div>
                    <div class="detail-value">${seHtml}</div>
                </div>` : ''}
                ${a.monitoring?.length ? `
                <div class="detail-row">
                    <div class="detail-label">Monitoring</div>
                    <div class="detail-value">
                        <ul class="detail-list">
                            ${a.monitoring.map(m => `<li>${m}</li>`).join('')}
                        </ul>
                    </div>
                </div>` : ''}
            </div>
        </div>`;
    }).join('');

    lucide.createIcons();
}

function toggleDetail(index) {
    const card = document.getElementById(`detail-${index}`);
    card.classList.toggle('expanded');
}

function renderMLChart(assessments) {
    const section = document.getElementById('mlSection');
    const hasML = assessments.some(a => a.ml_prediction?.available);

    if (!hasML) {
        section.classList.add('hidden');
        return;
    }
    section.classList.remove('hidden');

    const ctx = document.getElementById('mlChart').getContext('2d');

    // Destroy old chart if exists
    if (window._mlChart) window._mlChart.destroy();

    const labels = assessments.map(a => a.drug_name);
    const probData = {};

    // Collect probability distributions
    assessments.forEach(a => {
        if (a.ml_prediction?.probabilities) {
            Object.entries(a.ml_prediction.probabilities).forEach(([label, val]) => {
                if (!probData[label]) probData[label] = [];
                probData[label].push(val);
            });
        }
    });

    const colorMap = {
        'Low Risk': '#22c55e',
        'Moderate': '#f59e0b',
        'High Risk': '#ef4444',
        'Critical': '#dc2626',
    };

    const datasets = Object.entries(probData).map(([label, values]) => ({
        label,
        data: values,
        backgroundColor: colorMap[label] || '#64748b',
        borderRadius: 3,
    }));

    window._mlChart = new Chart(ctx, {
        type: 'bar',
        data: { labels, datasets },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } }
                },
            },
            scales: {
                x: {
                    stacked: true,
                    ticks: { color: '#94a3b8', font: { family: 'Inter' } },
                    grid: { color: 'rgba(255,255,255,0.03)' },
                },
                y: {
                    stacked: true,
                    max: 100,
                    ticks: {
                        color: '#94a3b8',
                        font: { family: 'Inter' },
                        callback: v => v + '%'
                    },
                    grid: { color: 'rgba(255,255,255,0.03)' },
                },
            },
        },
    });
}

// ── Patient Wizard ────────────────────────────────────────────────────────

const WIZARD_STEPS = [
    { title: 'About You', subtitle: 'Basic info to personalize your results' },
    { title: 'Health Conditions', subtitle: 'Select any conditions you have been diagnosed with' },
    { title: 'Health Numbers', subtitle: 'Enter any lab values you know (optional)' },
    { title: 'Current Medications', subtitle: 'What medications do you take now?' },
    { title: 'Check Medications', subtitle: 'Which medications would you like to check?' },
    { title: 'Your Results', subtitle: 'Here is your personalized assessment' },
];

const PLAIN_CONDITIONS = {
    'Diabetes': 'Diabetes (high blood sugar)',
    'Hypertension': 'High blood pressure',
    'CKD (Chronic Kidney Disease)': 'Kidney problems',
    'Heart Failure': 'Heart failure / weak heart',
    'Asthma': 'Asthma or wheezing',
    'COPD': 'Breathing problems (COPD)',
    'Liver Disease': 'Liver problems',
    'Peptic Ulcer': 'Stomach ulcer',
};

function renderPatientWizard() {
    const progressEl = document.getElementById('wizardProgress');
    const contentEl = document.getElementById('wizardContent');
    const navEl = document.getElementById('wizardNav');

    // Progress dots
    progressEl.innerHTML = WIZARD_STEPS.map((_, i) => {
        const cls = i + 1 < patientWizardStep ? 'done' : (i + 1 === patientWizardStep ? 'active' : '');
        return `<div class="wizard-step-dot ${cls}"></div>`;
    }).join('');

    // Step content
    const step = WIZARD_STEPS[patientWizardStep - 1];
    let html = `<div class="wizard-title">${step.title}</div><div class="wizard-subtitle">${step.subtitle}</div>`;

    switch (patientWizardStep) {
        case 1:
            html += `
                <div class="form-group"><label>Your first name (optional)</label>
                    <input type="text" id="wName" value="${patientData.name || ''}" placeholder="e.g. Sarah" /></div>
                <div class="form-row two-col">
                    <div class="form-group"><label>Age</label>
                        <input type="number" id="wAge" value="${patientData.age || 45}" min="1" max="120" /></div>
                    <div class="form-group"><label>Sex</label>
                        <select id="wSex">
                            <option value="F" ${patientData.sex === 'F' || !patientData.sex ? 'selected' : ''}>Female</option>
                            <option value="M" ${patientData.sex === 'M' ? 'selected' : ''}>Male</option>
                        </select></div>
                </div>
                <div class="form-group"><label>Weight (kg)</label>
                    <input type="number" id="wWeight" value="${patientData.weight || 70}" min="20" max="300" step="0.5" /></div>`;
            break;
        case 2:
            html += '<div class="checkbox-grid">';
            Object.entries(PLAIN_CONDITIONS).forEach(([key, label]) => {
                const checked = (patientData.conditions || []).includes(key) ? 'checked' : '';
                html += `<label class="check-item"><input type="checkbox" value="${key}" ${checked}><span>${label}</span></label>`;
            });
            html += '</div>';
            break;
        case 3:
            html += `
                <div class="form-row two-col">
                    <div class="form-group"><label>Kidney function (eGFR)</label>
                        <input type="number" id="wEgfr" value="${patientData.egfr || 90}" min="0" max="150" /></div>
                    <div class="form-group"><label>Blood sugar (HbA1c %)</label>
                        <input type="number" id="wHba1c" value="${patientData.hba1c || 5.5}" min="3" max="20" step="0.1" /></div>
                </div>
                <div class="form-row two-col">
                    <div class="form-group"><label>Blood pressure (top number)</label>
                        <input type="number" id="wSbp" value="${patientData.sbp || 120}" min="60" max="260" /></div>
                    <div class="form-group"><label>Cholesterol (LDL)</label>
                        <input type="number" id="wLdl" value="${patientData.ldl || 3.0}" min="0" max="15" step="0.1" /></div>
                </div>`;
            break;
        case 4:
            html += '<div class="checkbox-grid">';
            drugCatalog.forEach(d => {
                const checked = (patientData.currentMeds || []).includes(d.name) ? 'checked' : '';
                html += `<label class="check-item"><input type="checkbox" value="${d.name}" ${checked}><span>${d.name}</span></label>`;
            });
            html += '</div>';
            break;
        case 5:
            html += '<div class="checkbox-grid">';
            drugCatalog.forEach(d => {
                const checked = (patientData.checkDrugs || []).includes(d.name) ? 'checked' : '';
                html += `<label class="check-item"><input type="checkbox" value="${d.name}" ${checked}><span>${d.name}</span></label>`;
            });
            html += '</div>';
            break;
        case 6:
            html += '<div id="patientResults"><div class="skeleton" style="height:200px"></div></div>';
            break;
    }
    contentEl.innerHTML = html;

    // Navigation
    let navHtml = '';
    if (patientWizardStep > 1 && patientWizardStep < 6) {
        navHtml += '<button class="btn-secondary" onclick="wizardBack()"><i data-lucide="arrow-left"></i> Back</button>';
    } else {
        navHtml += '<div></div>';
    }
    if (patientWizardStep < 5) {
        navHtml += '<button class="btn-primary" style="width:auto;padding:10px 24px" onclick="wizardNext()">Next <i data-lucide="arrow-right"></i></button>';
    } else if (patientWizardStep === 5) {
        navHtml += '<button class="btn-primary" style="width:auto;padding:10px 24px" onclick="wizardNext()">See Results <i data-lucide="arrow-right"></i></button>';
    } else {
        navHtml += '<button class="btn-secondary" onclick="wizardRestart()"><i data-lucide="rotate-ccw"></i> Start Over</button>';
    }
    navEl.innerHTML = navHtml;
    lucide.createIcons();

    if (patientWizardStep === 6) runPatientAssessment();
}

function wizardSaveStep() {
    switch (patientWizardStep) {
        case 1:
            patientData.name = document.getElementById('wName')?.value || '';
            patientData.age = parseInt(document.getElementById('wAge')?.value) || 45;
            patientData.sex = document.getElementById('wSex')?.value || 'F';
            patientData.weight = parseFloat(document.getElementById('wWeight')?.value) || 70;
            break;
        case 2:
            patientData.conditions = [];
            document.querySelectorAll('#wizardContent input:checked').forEach(cb => patientData.conditions.push(cb.value));
            break;
        case 3:
            patientData.egfr = parseFloat(document.getElementById('wEgfr')?.value) || 90;
            patientData.hba1c = parseFloat(document.getElementById('wHba1c')?.value) || 5.5;
            patientData.sbp = parseInt(document.getElementById('wSbp')?.value) || 120;
            patientData.ldl = parseFloat(document.getElementById('wLdl')?.value) || 3.0;
            break;
        case 4:
            patientData.currentMeds = [];
            document.querySelectorAll('#wizardContent input:checked').forEach(cb => patientData.currentMeds.push(cb.value));
            break;
        case 5:
            patientData.checkDrugs = [];
            document.querySelectorAll('#wizardContent input:checked').forEach(cb => patientData.checkDrugs.push(cb.value));
            break;
    }
}

function wizardNext() {
    wizardSaveStep();
    if (patientWizardStep === 5 && !(patientData.checkDrugs?.length)) {
        alert('Please select at least one medication to check.');
        return;
    }
    patientWizardStep++;
    renderPatientWizard();
}

function wizardBack() {
    wizardSaveStep();
    patientWizardStep--;
    renderPatientWizard();
}

function wizardRestart() {
    patientData = {};
    patientWizardStep = 1;
    renderPatientWizard();
}

async function runPatientAssessment() {
    const patient = {
        name: patientData.name || 'Patient',
        age: patientData.age || 45,
        sex: patientData.sex || 'F',
        weight_kg: patientData.weight || 70,
        conditions: patientData.conditions || [],
        egfr: patientData.egfr || 90,
        alt: 25, ast: 25,
        hba1c: patientData.hba1c || 5.5,
        systolic_bp: patientData.sbp || 120,
        diastolic_bp: Math.round((patientData.sbp || 120) * 0.65),
        ldl: patientData.ldl || 3.0,
        inr: 1.0,
        current_meds: patientData.currentMeds || [],
        allergies: [],
    };

    try {
        const resp = await fetch(`${API_BASE}/api/assess`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ patient, drugs: patientData.checkDrugs || [] }),
        });

        if (!resp.ok) throw new Error();
        const data = await resp.json();
        renderPatientResults(data);
    } catch {
        document.getElementById('patientResults').innerHTML = `
            <div style="text-align:center;padding:24px;color:var(--text-muted)">
                Could not reach the server. Please ensure the backend is running.
            </div>`;
    }
}

function renderPatientResults(data) {
    const container = document.getElementById('patientResults');
    const assessments = data.assessments || [];
    const interactions = data.interactions || [];

    const suitMap = {
        'Suitable': { label: 'Good Match', cls: 'safe', desc: 'This medication is suitable based on your health profile.' },
        'Caution': { label: 'Use With Caution', cls: 'caution', desc: 'May be appropriate, but your doctor should review the dose carefully.' },
        'Avoid': { label: 'Not Recommended', cls: 'avoid', desc: 'This medication may not be safe for you. Talk to your doctor.' },
        'Contraindicated': { label: 'Do Not Use', cls: 'critical', desc: 'This medication could be harmful. Avoid unless directed by your doctor.' },
    };

    let html = assessments.map(a => {
        const info = suitMap[a.suitability] || suitMap['Caution'];
        return `
            <div class="patient-result ${info.cls} fade-in">
                <div class="patient-result-header">
                    <div class="patient-result-drug">${a.drug_name}</div>
                    <span class="risk-badge ${info.cls}"><span class="risk-dot ${info.cls}"></span>${info.label}</span>
                </div>
                <div class="patient-result-desc">${info.desc}</div>
                ${a.reasons?.length ? `<div class="patient-result-desc" style="margin-top:8px">${a.reasons[0]}</div>` : ''}
            </div>`;
    }).join('');

    if (interactions.length) {
        html += '<div style="margin-top:16px">';
        html += '<div class="section-label" style="margin-bottom:8px">Interactions to discuss with your doctor</div>';
        interactions.forEach(ix => {
            html += `<div class="ddi-item"><div class="ddi-drugs">${ix.drug_a} + ${ix.drug_b}</div><div>${ix.message}</div></div>`;
        });
        html += '</div>';
    }

    container.innerHTML = html;
}

// ── Utils ─────────────────────────────────────────────────────────────────

function resetForm() {
    document.getElementById('patientForm').reset();
    document.getElementById('patientAge').value = 50;
    document.getElementById('patientWeight').value = 70;
    document.getElementById('labEgfr').value = 90;
    document.getElementById('labAlt').value = 25;
    document.getElementById('labHba1c').value = 5.5;
    document.getElementById('labSbp').value = 120;
    document.getElementById('labLdl').value = 3.0;
    document.getElementById('labInr').value = 1.0;
    selectedDrugs.clear();
    renderDrugGrid();
    document.getElementById('resultsCard').classList.add('hidden');
}
