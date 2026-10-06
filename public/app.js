// RETENTIX AI | Frontend Logic & Dynamic Client Controller

const API_BASE = '/api';

// Customer Archetype Presets
const PRESETS = {
  highRisk: {
    gender: 'Female', SeniorCitizen: 0, Partner: 'No', Dependents: 'No',
    tenure: 2, PhoneService: 'Yes', MultipleLines: 'No',
    InternetService: 'Fiber optic', OnlineSecurity: 'No', OnlineBackup: 'No',
    DeviceProtection: 'No', TechSupport: 'No', StreamingTV: 'Yes',
    StreamingMovies: 'Yes', Contract: 'Month-to-month', PaperlessBilling: 'Yes',
    PaymentMethod: 'Electronic check', MonthlyCharges: 92.50, TotalCharges: 185.00
  },
  loyalVeteran: {
    gender: 'Male', SeniorCitizen: 0, Partner: 'Yes', Dependents: 'Yes',
    tenure: 60, PhoneService: 'Yes', MultipleLines: 'Yes',
    InternetService: 'DSL', OnlineSecurity: 'Yes', OnlineBackup: 'Yes',
    DeviceProtection: 'Yes', TechSupport: 'Yes', StreamingTV: 'No',
    StreamingMovies: 'No', Contract: 'Two year', PaperlessBilling: 'No',
    PaymentMethod: 'Credit card (automatic)', MonthlyCharges: 64.00, TotalCharges: 3840.00
  },
  vulnerableFamily: {
    gender: 'Female', SeniorCitizen: 0, Partner: 'Yes', Dependents: 'Yes',
    tenure: 14, PhoneService: 'Yes', MultipleLines: 'Yes',
    InternetService: 'Fiber optic', OnlineSecurity: 'No', OnlineBackup: 'Yes',
    DeviceProtection: 'No', TechSupport: 'No', StreamingTV: 'Yes',
    StreamingMovies: 'Yes', Contract: 'Month-to-month', PaperlessBilling: 'Yes',
    PaymentMethod: 'Electronic check', MonthlyCharges: 98.75, TotalCharges: 1382.50
  },
  techEnthusiast: {
    gender: 'Male', SeniorCitizen: 0, Partner: 'No', Dependents: 'No',
    tenure: 36, PhoneService: 'Yes', MultipleLines: 'Yes',
    InternetService: 'Fiber optic', OnlineSecurity: 'Yes', OnlineBackup: 'Yes',
    DeviceProtection: 'Yes', TechSupport: 'Yes', StreamingTV: 'Yes',
    StreamingMovies: 'Yes', Contract: 'One year', PaperlessBilling: 'Yes',
    PaymentMethod: 'Bank transfer (automatic)', MonthlyCharges: 112.50, TotalCharges: 4050.00
  }
};

let scoredBatchResults = [];

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initPresets();
  initPredictionForm();
  initBatchInference();
  initRoiCalculator();
  checkApiHealth();
});

// Navigation Handling
function initNavigation() {
  const navItems = document.querySelectorAll('.nav-item');
  navItems.forEach(item => {
    item.addEventListener('click', () => {
      const tabId = item.dataset.tab;
      
      navItems.forEach(i => i.classList.remove('active'));
      item.classList.add('active');
      
      document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
      });
      const activeTab = document.getElementById(tabId);
      if (activeTab) activeTab.classList.add('active');
    });
  });
}

// Preset Loader
function initPresets() {
  document.querySelectorAll('.btn-preset').forEach(btn => {
    btn.addEventListener('click', () => {
      const presetKey = btn.dataset.preset;
      const data = PRESETS[presetKey];
      if (!data) return;

      Object.keys(data).forEach(key => {
        const input = document.getElementById(`field_${key}`);
        if (input) {
          input.value = data[key];
        }
      });
      
      // Auto-trigger prediction
      predictSingle();
    });
  });
}

// Single Customer Prediction Form
function initPredictionForm() {
  const form = document.getElementById('singlePredictForm');
  if (!form) return;

  // Auto calculate total charges on tenure/monthly change
  const tenureInput = document.getElementById('field_tenure');
  const monthlyInput = document.getElementById('field_MonthlyCharges');
  const totalInput = document.getElementById('field_TotalCharges');

  function updateTotal() {
    const t = parseFloat(tenureInput.value) || 0;
    const m = parseFloat(monthlyInput.value) || 0;
    totalInput.value = (t * m).toFixed(2);
  }

  tenureInput.addEventListener('input', updateTotal);
  monthlyInput.addEventListener('input', updateTotal);

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    predictSingle();
  });

  // Run initial prediction with default preset
  predictSingle();
}

async function predictSingle() {
  const form = document.getElementById('singlePredictForm');
  const formData = new FormData(form);
  const payload = {};
  
  formData.forEach((val, key) => {
    if (['SeniorCitizen', 'tenure'].includes(key)) {
      payload[key] = parseInt(val, 10);
    } else if (['MonthlyCharges', 'TotalCharges'].includes(key)) {
      payload[key] = parseFloat(val);
    } else {
      payload[key] = val;
    }
  });

  const btn = document.getElementById('btnSubmitPredict');
  if (btn) btn.innerHTML = '<span>⚡</span> Analyzing Probability...';

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Prediction failed');

    renderPredictionResult(data);
  } catch (err) {
    alert(`Inference Error: ${err.message}`);
  } finally {
    if (btn) btn.innerHTML = '<span>⚡</span> Run AI Risk Assessment';
  }
}

function renderPredictionResult(data) {
  const probElem = document.getElementById('resProbability');
  const badgeElem = document.getElementById('resBadge');
  const progressElem = document.getElementById('resProgressBar');
  const bundleElem = document.getElementById('resBundle');
  const retainElem = document.getElementById('resRetainProb');
  const interventionsList = document.getElementById('interventionsList');

  const prob = data.churn_probability;
  probElem.textContent = `${prob.toFixed(1)}%`;
  probElem.style.color = data.risk_color;
  
  badgeElem.textContent = data.risk_level;
  badgeElem.className = `risk-badge ${data.badge_class}`;

  if (progressElem) {
    progressElem.style.width = `${prob}%`;
    progressElem.style.backgroundColor = data.risk_color;
  }

  if (retainElem) {
    retainElem.textContent = `${data.no_churn_probability.toFixed(1)}%`;
  }

  if (bundleElem) {
    bundleElem.textContent = `Model: ${data.model_name} (Bundle ${data.active_bundle})`;
  }

  // Interventions
  if (interventionsList && data.interventions) {
    interventionsList.innerHTML = data.interventions.map(act => `
      <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 14px; margin-bottom: 10px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 4px;">
          <div style="font-weight:700; color:#fff; font-size:0.9rem;">${act.action}</div>
          <span style="font-size:0.7rem; font-weight:700; padding:2px 8px; border-radius:999px; background:rgba(99,102,241,0.2); color:#A5B4FC;">${act.priority}</span>
        </div>
        <div style="font-size:0.82rem; color:#94A3B8;">${act.description}</div>
        <div style="font-size:0.75rem; color:#10B981; font-weight:600; margin-top:4px;">✨ ${act.impact}</div>
      </div>
    `).join('');
  }
}

// Batch Portfolio Prediction
function initBatchInference() {
  const fileInput = document.getElementById('batchFileInput');
  const dropzone = document.getElementById('batchDropzone');
  const demoBtn = document.getElementById('btnLoadDemoBatch');
  const exportBtn = document.getElementById('btnExportBatchCsv');

  if (dropzone && fileInput) {
    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.style.borderColor = '#6366F1'; });
    dropzone.addEventListener('dragleave', () => { dropzone.style.borderColor = ''; });
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.style.borderColor = '';
      if (e.dataTransfer.files.length) {
        processBatchFile(e.dataTransfer.files[0]);
      }
    });
    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length) {
        processBatchFile(e.target.files[0]);
      }
    });
  }

  if (demoBtn) {
    demoBtn.addEventListener('click', async () => {
      demoBtn.disabled = true;
      demoBtn.textContent = 'Loading demo...';
      try {
        const res = await fetch(`${API_BASE}/sample_csv`);
        const csvText = await res.text();
        sendBatchCsv(csvText);
      } catch (err) {
        alert('Failed to load demo CSV: ' + err.message);
      } finally {
        demoBtn.disabled = false;
        demoBtn.textContent = '📥 Load Demo Sample Batch';
      }
    });
  }

  if (exportBtn) {
    exportBtn.addEventListener('click', () => {
      if (!scoredBatchResults.length) return;
      downloadBatchCsv(scoredBatchResults);
    });
  }
}

function processBatchFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    sendBatchCsv(e.target.result);
  };
  reader.readAsText(file);
}

async function sendBatchCsv(csvText) {
  const statusElem = document.getElementById('batchStatus');
  if (statusElem) statusElem.textContent = '⚡ Running batch scoring against active pipeline bundle...';

  try {
    const res = await fetch(`${API_BASE}/predict_batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'text/csv' },
      body: csvText
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Batch scoring failed');

    scoredBatchResults = data.records;
    renderBatchResults(data);
    if (statusElem) statusElem.textContent = `✅ Successfully scored ${data.total_records} customer records.`;
  } catch (err) {
    if (statusElem) statusElem.textContent = `❌ Error: ${err.message}`;
  }
}

function renderBatchResults(data) {
  document.getElementById('batchStatsContainer').style.display = 'grid';
  document.getElementById('batchResultsTableContainer').style.display = 'block';

  document.getElementById('statTotalBatch').textContent = data.total_records;
  document.getElementById('statChurnCount').textContent = data.churn_count;
  document.getElementById('statBatchChurnRate').textContent = `${data.churn_rate_pct}%`;
  document.getElementById('statHighRiskCount').textContent = data.critical_high_risk_count;

  const tbody = document.getElementById('batchTableBody');
  tbody.innerHTML = data.records.map(r => `
    <tr>
      <td><strong>${r.customerID}</strong></td>
      <td><span class="risk-badge" style="background:${r.churn ? 'rgba(239,68,68,0.15)' : 'rgba(16,185,129,0.15)'}; color:${r.churn ? '#F87171' : '#34D399'}; font-size:0.75rem; padding:4px 10px;">${r.prediction}</span></td>
      <td><strong style="color:${r.risk_color};">${r.churn_probability}%</strong></td>
      <td>${r.risk_level}</td>
      <td>${r.Contract || '-'}</td>
      <td>${r.tenure ? r.tenure + ' mos' : '-'}</td>
      <td>${r.MonthlyCharges ? '$' + r.MonthlyCharges : '-'}</td>
    </tr>
  `).join('');
}

function downloadBatchCsv(records) {
  const headers = ['customerID', 'prediction', 'churn_probability', 'risk_level', 'Contract', 'tenure', 'MonthlyCharges'];
  const csvRows = [headers.join(',')];

  records.forEach(r => {
    const row = headers.map(h => `"${r[h] !== undefined ? r[h] : ''}"`);
    csvRows.push(row.join(','));
  });

  const blob = new Blob([csvRows.join('\n')], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `scored_churn_predictions_${Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

// Retention ROI Simulator
function initRoiCalculator() {
  const countSlider = document.getElementById('roiTargetCount');
  const successSlider = document.getElementById('roiSuccessRate');
  const countVal = document.getElementById('roiCountVal');
  const successVal = document.getElementById('roiSuccessVal');

  function calculate() {
    const count = parseInt(countSlider.value, 10);
    const rate = parseInt(successSlider.value, 10);
    
    countVal.textContent = count.toLocaleString();
    successVal.textContent = `${rate}%`;

    const avgArpu = 75; // average monthly revenue
    const annualArpu = avgArpu * 12;
    const savedCustomers = Math.round(count * (rate / 100));
    const annualSaved = savedCustomers * annualArpu;

    document.getElementById('roiSavedCount').textContent = savedCustomers.toLocaleString();
    document.getElementById('roiRevenueSaved').textContent = `$${annualSaved.toLocaleString()}`;
  }

  if (countSlider && successSlider) {
    countSlider.addEventListener('input', calculate);
    successSlider.addEventListener('input', calculate);
    calculate();
  }
}

// System Health & Active Bundle Badge
async function checkApiHealth() {
  const badge = document.getElementById('systemHealthBadge');
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    if (data.status === 'healthy') {
      badge.textContent = `Online • Bundle ${data.active_bundle} (${data.scikit_learn_version})`;
      badge.style.color = '#34D399';
    }
  } catch (err) {
    badge.textContent = 'API Initializing...';
    badge.style.color = '#FBBF24';
  }
}
