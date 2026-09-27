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
