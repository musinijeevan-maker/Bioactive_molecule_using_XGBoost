/* ============================================================
   BIOACTIVE MOLECULE PREDICTOR — script.js
   Vanilla JavaScript — connects frontend to FastAPI backend
   ============================================================ */

// ─── Dynamic Backend URL ──────────────────────────────────────────────────────
// If loaded from localhost:8000/app/, origin is used.
// If loaded via file:// protocol directly, defaults to http://127.0.0.1:8000
const API_BASE = (window.location.protocol === "file:")
  ? "http://127.0.0.1:8000"
  : window.location.origin;

// ─── Example SMILES to cycle through ─────────────────────────────────────────
const EXAMPLES = [
  "CCO",                                          // Ethanol
  "CC(=O)Oc1ccccc1C(=O)O",                       // Aspirin
  "c1ccccc1",                                     // Benzene
  "CC(C)Cc1ccc(cc1)C(C)C(=O)O",                  // Ibuprofen
  "CS(=O)(=O)c1ccc(-c2csc(CC(=O)O)c2-c2ccc(F)cc2)cc1"  // COX-2 active/inactive candidate
];
let exampleIndex = 0;

// ─── DOM Element References ───────────────────────────────────────────────────
const smilesInput    = document.getElementById("smilesInput");
const predictBtn     = document.getElementById("predictBtn");
const clearBtn       = document.getElementById("clearBtn");
const loadExBtn      = document.getElementById("loadExampleBtn");
const inputError     = document.getElementById("inputError");
const resultEmpty    = document.getElementById("resultEmpty");
const resultLoading  = document.getElementById("resultLoading");
const resultData     = document.getElementById("resultData");
const backendStatus  = document.getElementById("backendStatus");
const navToggle      = document.getElementById("navToggle");
const navMobile      = document.getElementById("navMobile");

// ─── 1. Navigation ────────────────────────────────────────────────────────────
if (navToggle) {
  navToggle.addEventListener("click", () => {
    const isOpen = navMobile.classList.toggle("open");
    navToggle.setAttribute("aria-expanded", String(isOpen));
  });
}

document.querySelectorAll(".mobile-link").forEach((link) => {
  link.addEventListener("click", () => {
    navMobile.classList.remove("open");
    navToggle.setAttribute("aria-expanded", "false");
  });
});

document.addEventListener("click", (e) => {
  if (navMobile && navToggle && !navMobile.contains(e.target) && !navToggle.contains(e.target)) {
    navMobile.classList.remove("open");
    navToggle.setAttribute("aria-expanded", "false");
  }
});

document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
  anchor.addEventListener("click", function (e) {
    const target = document.querySelector(this.getAttribute("href"));
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: "smooth" });
    }
  });
});

// ─── 2. Backend Health Check ──────────────────────────────────────────────────
async function checkBackendHealth() {
  if (!backendStatus) return;
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      backendStatus.textContent = "Online";
      backendStatus.style.color = "#16a34a";
    } else {
      backendStatus.textContent = "Server response error";
      backendStatus.style.color = "#d97706";
    }
  } catch {
    backendStatus.textContent = "Offline (run: uvicorn main:app --reload --port 8000)";
    backendStatus.style.color = "#dc2626";
  }
}

checkBackendHealth();

// ─── 3. Load Example ─────────────────────────────────────────────────────────
function loadExample() {
  smilesInput.value = EXAMPLES[exampleIndex % EXAMPLES.length];
  exampleIndex++;
  clearError();
  smilesInput.focus();
}

if (loadExBtn) {
  loadExBtn.addEventListener("click", loadExample);
}

// ─── 4. Input Validation ─────────────────────────────────────────────────────
function validateInput() {
  const raw = smilesInput.value.trim();

  if (raw === "") {
    showError("Please enter a SMILES string before predicting.");
    return false;
  }
  if (/^\d+$/.test(raw)) {
    showError("This doesn't look like a valid SMILES string. Example: CCO");
    return false;
  }
  if (/\s/.test(raw)) {
    showError("SMILES strings should not contain spaces. Please check your input.");
    return false;
  }
  clearError();
  return true;
}

function showError(msg) {
  inputError.textContent = msg;
  inputError.classList.add("visible");
  smilesInput.classList.add("input-error");
}

function clearError() {
  inputError.textContent = "";
  inputError.classList.remove("visible");
  smilesInput.classList.remove("input-error");
}

if (smilesInput) {
  smilesInput.addEventListener("input", clearError);
}

// ─── 5. Loading State ─────────────────────────────────────────────────────────
function setLoadingState(isLoading) {
  if (isLoading) {
    resultEmpty.style.display   = "none";
    resultLoading.style.display = "flex";
    resultData.style.display    = "none";
    predictBtn.disabled         = true;
    predictBtn.textContent      = "Predicting…";
  } else {
    resultLoading.style.display = "none";
    predictBtn.disabled         = false;
    predictBtn.textContent      = "Predict Bioactivity";
  }
}

// ─── 6. Display Result ────────────────────────────────────────────────────────
function displayPrediction(data) {
  const isActive    = String(data.prediction).toLowerCase() === "active";
  const badgeClass  = isActive ? "badge-active" : "badge-inactive";
  const label       = isActive ? "Active" : "Inactive";
  const note        = isActive ? "(Biologically Active)" : "(Biologically Inactive)";
  const probPercent = (data.probability * 100).toFixed(1) + "%";

  const displaySmiles = data.smiles.length > 40
    ? data.smiles.substring(0, 40) + "…"
    : data.smiles;

  resultData.innerHTML = `
    <div class="result-badge ${badgeClass}">
      ${label}
      <span style="font-size:0.85rem; font-weight:500; opacity:0.85;">${note}</span>
    </div>
    <div class="result-rows">
      <div class="result-row">
        <span class="result-row-label">Prediction</span>
        <span class="result-row-value">${label}</span>
      </div>
      <div class="result-row">
        <span class="result-row-label">Confidence Probability</span>
        <span class="result-row-value">${probPercent}</span>
      </div>
      <div class="result-row">
        <span class="result-row-label">Model Used</span>
        <span class="result-row-value">${data.model}</span>
      </div>
      <div class="result-row">
        <span class="result-row-label">Input SMILES</span>
        <span class="result-row-value">${displaySmiles}</span>
      </div>
    </div>
  `;

  resultData.style.display = "block";
}

// ─── 7. Display Error ─────────────────────────────────────────────────────────
function displayError(message) {
  resultData.innerHTML = `
    <div class="result-empty" style="color:var(--error);">
      <svg width="40" height="40" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="2" aria-hidden="true">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="15" y1="9" x2="9" y2="15"></line>
        <line x1="9"  y1="9" x2="15" y2="15"></line>
      </svg>
      <p style="margin-top:8px;">${message}</p>
    </div>
  `;
  resultData.style.display = "block";
}

// ─── 8. Clear ─────────────────────────────────────────────────────────────────
function clearPrediction() {
  smilesInput.value           = "";
  clearError();
  resultEmpty.style.display   = "flex";
  resultLoading.style.display = "none";
  resultData.style.display    = "none";
  resultData.innerHTML        = "";
  smilesInput.focus();
}

if (clearBtn) {
  clearBtn.addEventListener("click", clearPrediction);
}

// ─── 9. Predict — sends real request to FastAPI ───────────────────────────────
async function predictMolecule() {
  if (!validateInput()) return;

  const smiles = smilesInput.value.trim();
  setLoadingState(true);

  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ smiles: smiles })
    });

    const data = await response.json();

    if (!response.ok) {
      const errMsg = data.detail || "Prediction request rejected by backend.";
      setLoadingState(false);
      displayError(errMsg);
      return;
    }

    setLoadingState(false);
    displayPrediction(data);

  } catch (err) {
    setLoadingState(false);
    displayError(
      `Could not reach the backend at <strong>${API_BASE}</strong>.<br>` +
      `Ensure FastAPI server is running: <code>uvicorn main:app --reload --port 8000</code>`
    );
    console.error("Fetch error:", err);
  }
}

if (predictBtn) {
  predictBtn.addEventListener("click", predictMolecule);
}

if (smilesInput) {
  smilesInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.ctrlKey) {
      e.preventDefault();
      predictMolecule();
    }
  });
}

// ============================================================
// 10. INTERACTIVE DIAGNOSTICS STUDIO LOGIC & CHARTS
// ============================================================

// --- Studio Tabs Switching ---
const studioTabBtns = document.querySelectorAll('.studio-tab-btn');
const tabPanels = document.querySelectorAll('.tab-content-panel');

studioTabBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    const targetTabId = btn.getAttribute('data-tab');
    
    studioTabBtns.forEach(b => b.classList.remove('active'));
    tabPanels.forEach(p => p.classList.remove('active'));
    
    btn.classList.add('active');
    const targetPanel = document.getElementById(targetTabId);
    if (targetPanel) {
      targetPanel.classList.add('active');
    }

    // Trigger chart render on tab switch if needed
    if (targetTabId === 'tab-roc' && !rocChartInstance) {
      initRocChart();
    } else if (targetTabId === 'tab-benchmark' && !benchmarkChartInstance) {
      initBenchmarkChart();
    } else if (targetTabId === 'tab-radar' && !radarChartInstance) {
      initRadarChart();
    }
  });
});

// --- Confusion Matrix Raw Count vs Percentage Toggle ---
const cmCountBtn = document.getElementById('cmCountBtn');
const cmPercentBtn = document.getElementById('cmPercentBtn');
const cmCells = [
  document.getElementById('cmTN'),
  document.getElementById('cmFP'),
  document.getElementById('cmFN'),
  document.getElementById('cmTP')
];

if (cmCountBtn && cmPercentBtn) {
  cmCountBtn.addEventListener('click', () => {
    cmCountBtn.classList.add('active');
    cmPercentBtn.classList.remove('active');
    cmCells.forEach(cell => {
      if (cell) cell.textContent = cell.getAttribute('data-raw');
    });
  });

  cmPercentBtn.addEventListener('click', () => {
    cmPercentBtn.classList.add('active');
    cmCountBtn.classList.remove('active');
    cmCells.forEach(cell => {
      if (cell) cell.textContent = cell.getAttribute('data-pct');
    });
  });
}

// --- Chart Instances ---
let rocChartInstance = null;
let benchmarkChartInstance = null;
let radarChartInstance = null;

// 1. Dynamic ROC Curve
function initRocChart() {
  const canvas = document.getElementById('rocChartCanvas');
  if (!canvas || typeof Chart === 'undefined') return;

  const ctx = canvas.getContext('2d');
  rocChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'],
      datasets: [
        {
          label: 'XGBoost (AUC = 0.8699)',
          data: [
            {x: 0.0, y: 0.0},
            {x: 0.04, y: 0.46},
            {x: 0.08, y: 0.65},
            {x: 0.15, y: 0.78},
            {x: 0.25, y: 0.88},
            {x: 0.39, y: 0.912},
            {x: 0.55, y: 0.96},
            {x: 0.75, y: 0.985},
            {x: 1.0, y: 1.0}
          ],
          borderColor: '#1a6fdb',
          backgroundColor: 'rgba(26, 111, 219, 0.12)',
          fill: true,
          tension: 0.35,
          borderWidth: 3,
          pointRadius: 4,
          pointHoverRadius: 7
        },
        {
          label: 'Random Forest (AUC = 0.8639)',
          data: [
            {x: 0.0, y: 0.0},
            {x: 0.06, y: 0.42},
            {x: 0.12, y: 0.62},
            {x: 0.20, y: 0.75},
            {x: 0.32, y: 0.86},
            {x: 0.44, y: 0.918},
            {x: 0.60, y: 0.95},
            {x: 0.80, y: 0.98},
            {x: 1.0, y: 1.0}
          ],
          borderColor: '#0d9488',
          borderDash: [6, 4],
          backgroundColor: 'transparent',
          fill: false,
          tension: 0.35,
          borderWidth: 2.5,
          pointRadius: 3,
          pointHoverRadius: 6
        },
        {
          label: 'Random Chance (AUC = 0.50)',
          data: [{x: 0, y: 0}, {x: 1, y: 1}],
          borderColor: '#94a3b8',
          borderDash: [4, 4],
          borderWidth: 1.5,
          pointRadius: 0,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 1200,
        easing: 'easeOutQuart'
      },
      scales: {
        x: {
          type: 'linear',
          min: 0,
          max: 1,
          title: {
            display: true,
            text: 'False Positive Rate (1 - Specificity)',
            font: { weight: 'bold', size: 12 }
          },
          grid: { color: 'rgba(0,0,0,0.05)' }
        },
        y: {
          min: 0,
          max: 1,
          title: {
            display: true,
            text: 'True Positive Rate (Sensitivity / Recall)',
            font: { weight: 'bold', size: 12 }
          },
          grid: { color: 'rgba(0,0,0,0.05)' }
        }
      },
      plugins: {
        legend: {
          position: 'bottom',
          labels: { boxWidth: 14, font: { weight: '600' } }
        },
        tooltip: {
          callbacks: {
            label: function(ctx) {
              return ${ctx.dataset.label}: TPR=%, FPR=%;
            }
          }
        }
      }
    }
  });
}

// 2. Multi-Model Benchmark Chart
function initBenchmarkChart() {
  const canvas = document.getElementById('benchmarkChartCanvas');
  if (!canvas || typeof Chart === 'undefined') return;

  const ctx = canvas.getContext('2d');
  const models = ['XGBoost', 'Random Forest', 'Linear SVM', 'RBF Network', 'Naive Bayes'];
  
  const allDatasets = [
    {
      label: 'Holdout F1-Score',
      data: [0.8869, 0.8824, 0.8548, 0.8156, 0.5618],
      backgroundColor: '#1a6fdb',
      borderRadius: 6
    },
    {
      label: 'Accuracy',
      data: [0.8302, 0.8214, 0.7877, 0.7006, 0.5313],
      backgroundColor: '#0d9488',
      borderRadius: 6
    },
    {
      label: 'ROC-AUC',
      data: [0.8699, 0.8639, 0.7843, 0.5810, 0.7544],
      backgroundColor: '#6366f1',
      borderRadius: 6
    }
  ];

  benchmarkChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: models,
      datasets: allDatasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 1000,
        easing: 'easeOutQuart'
      },
      scales: {
        y: {
          min: 0,
          max: 1.0,
          ticks: {
            callback: value => (value * 100) + '%'
          },
          grid: { color: 'rgba(0,0,0,0.05)' }
        },
        x: {
          grid: { display: false },
          ticks: { font: { weight: '600' } }
        }
      },
      plugins: {
        legend: {
          position: 'bottom',
          labels: { boxWidth: 14, font: { weight: '600' } }
        }
      }
    }
  });

  // Filter Chips Handler
  const chips = document.querySelectorAll('.chart-chip');
  chips.forEach(chip => {
    chip.addEventListener('click', () => {
      chips.forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      const metric = chip.getAttribute('data-metric');

      if (metric === 'all') {
        benchmarkChartInstance.data.datasets = allDatasets;
      } else if (metric === 'f1') {
        benchmarkChartInstance.data.datasets = [allDatasets[0]];
      } else if (metric === 'acc') {
        benchmarkChartInstance.data.datasets = [allDatasets[1]];
      } else if (metric === 'auc') {
        benchmarkChartInstance.data.datasets = [allDatasets[2]];
      }
      benchmarkChartInstance.update();
    });
  });
}

// 3. Multi-Metric Spider Radar Chart
function initRadarChart() {
  const canvas = document.getElementById('radarChartCanvas');
  if (!canvas || typeof Chart === 'undefined') return;

  const ctx = canvas.getContext('2d');
  radarChartInstance = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: ['Accuracy', 'Precision', 'Recall (Sensitivity)', 'Specificity', 'Holdout F1', 'ROC-AUC'],
      datasets: [
        {
          label: 'XGBoost (Winner)',
          data: [0.8302, 0.8636, 0.9116, 0.6098, 0.8869, 0.8699],
          borderColor: '#1a6fdb',
          backgroundColor: 'rgba(26, 111, 219, 0.25)',
          borderWidth: 2.5,
          pointBackgroundColor: '#1a6fdb',
          pointRadius: 4
        },
        {
          label: 'Random Forest',
          data: [0.8214, 0.8499, 0.9176, 0.5608, 0.8824, 0.8639],
          borderColor: '#0d9488',
          backgroundColor: 'rgba(13, 148, 136, 0.15)',
          borderWidth: 2,
          pointBackgroundColor: '#0d9488',
          pointRadius: 3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 1200,
        easing: 'easeOutQuart'
      },
      scales: {
        r: {
          min: 0,
          max: 1.0,
          ticks: {
            stepSize: 0.2,
            callback: value => (value * 100) + '%'
          },
          pointLabels: {
            font: { size: 12, weight: '600' }
          }
        }
      },
      plugins: {
        legend: {
          position: 'bottom',
          labels: { boxWidth: 14, font: { weight: '600' } }
        }
      }
    }
  });
}

// Auto-initialize charts on DOM load
document.addEventListener('DOMContentLoaded', () => {
  // Pre-load active tab chart if needed
});
