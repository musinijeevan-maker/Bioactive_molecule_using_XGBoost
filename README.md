# Bioactive Molecule Predictor

> **Academic College Project** — B.Tech CSE  
> **Domain**: Drug Discovery &middot; **Task**: Molecular Bioactivity Classification  
> **Target Enzyme**: Cyclooxygenase-2 (COX-2)

---

## 1. Quick Access Links

With the FastAPI server running (`http://127.0.0.1:8000`):

* 🌐 **Interactive Web App**: [http://127.0.0.1:8000/app/](http://127.0.0.1:8000/app/)
* 📚 **Interactive Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* 🩺 **Backend Health Status**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 2. Project Architecture & Flow

```
                      USER / BROWSER
                            ↓ (enters SMILES string)
                 FRONTEND (HTML + CSS + Vanilla JS)
                            ↓ fetch("http://127.0.0.1:8000/predict")
                    FASTAPI BACKEND (Python)
                            ↓ Pydantic Input Validation
                 RDKit MOLECULAR CONVERSION
                            ↓ Morgan Circular Fingerprint (radius=2, 2048-bit)
                  SAVED BEST MODEL (best_model.pkl)
                            ↓ model.predict() & predict_proba()
                     JSON API RESPONSE
                            ↓ { "prediction": "active", "probability": 0.97, "model": "XGBoost" }
                 DYNAMIC DOM RESULT DISPLAY
                            ↓ (Active / Inactive Badge + Confidence %)
                          USER
```

---

## 3. Dataset & Data Leakage Prevention

* **File**: `CHEMBL230_Preprocessed_Data.csv` (ChEMBL bioactivity records for COX-2).
* **Samples**: 6,839 compounds.
* **Columns**:
  * `molecule_chembl_id`: Molecule ID in ChEMBL.
  * `smiles`: Molecular structure line notation.
  * `standard_value`: Experimental $IC_{50}$ in nanomolar ($nM$).
  * `bioactivity_class`: Target label (`active` vs `inactive`).
* **Critical Preprocessing Rule — Data Leakage Prevention**:
  `standard_value` was the exact numerical assay value used to create the `bioactivity_class` label. Therefore, **`standard_value` is strictly excluded from the ML feature matrix**. The machine learning models must learn directly from **chemical structural features (SMILES)**, not from the target's source measurement.

---

## 4. Experimental Setup: 70/30 Split + 10-Fold Cross-Validation

Following rigorous academic research standards:
1. **70% Development Set** (4,787 molecules): Used to perform **10-Fold Stratified Cross-Validation** across all 5 candidate models to evaluate stability and generalization.
2. **Untouched 30% Holdout Set** (2,052 molecules): Reserved strictly for final evaluation. It is never seen during 10-fold CV or training.
3. **Automatic Best Model Selection Rule**:
   $$\text{Primary: Highest Holdout F1-Score} \longrightarrow \text{Tie-breaker: Highest ROC-AUC}$$

---

## 5. Real Experimental Benchmark Results

| Model | Mean 10-Fold CV F1 | Holdout Accuracy | Precision | Recall | Specificity | Holdout F1-Score | ROC-AUC | Status |
|---|---|---|---|---|---|---|---|---|
| **XGBoost** | 0.8853 | **82.55%** | 0.8400 | **0.9407** | 0.5118 | **0.8875** | **0.8677** | 🏆 **Auto-Selected Winner** |
| **Random Forest** | **0.8936** | 82.36% | **0.8458** | 0.9280 | 0.5390 | 0.8850 | 0.8620 | Ensemble Baseline |
| **Linear SVM** | 0.8585 | 78.41% | 0.8550 | 0.8488 | 0.6080 | 0.8519 | 0.7867 | Linear Baseline |
| **RBF Network** | 0.8168 | 71.64% | 0.7407 | 0.9420 | 0.1016 | 0.8293 | 0.5975 | Neural Baseline |
| **Naive Bayes** | 0.5221 | 52.63% | 0.8773 | 0.4097 | **0.8439** | 0.5586 | 0.7401 | Probabilistic Baseline |

* **Winner**: **XGBoost** achieved the highest Holdout F1 (`0.8875`), highest Holdout Accuracy (`82.55%`), and highest ROC-AUC (`0.8677`).
* The system automatically persisted `backend/model/best_model.pkl` and `backend/model/metadata.json`.

---

## 6. Project Structure

```
d:/Bioactive molecule prediction/
├── CHEMBL230_Preprocessed_Data.csv   # Preprocessed ChEMBL dataset
├── ML_Pipeline.ipynb                 # Interactive ML pipeline notebook
├── train_model.py                   # 10-fold CV & model evaluation pipeline
├── README.md                        # Documentation & mentor preparation
├── backend/
│   ├── main.py                      # FastAPI REST application
│   └── model/
│       ├── best_model.pkl           # Saved winning model artifact
│       └── metadata.json            # Model parameters, metrics & labels
└── frontend/
    ├── index.html                   # Semantic scientific UI
    ├── style.css                    # Professional responsive stylesheet
    └── script.js                    # Fetch client connecting to FastAPI
```

---

## 7. How to Run the Application

### Step 1: Run Training (Optional — already trained & saved)
```powershell
python train_model.py
```

### Step 2: Start FastAPI Server
```powershell
cd "d:\Bioactive molecule prediction\backend"
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### Step 3: Open and Use the Web Interface
Open your web browser and go to:
**[http://127.0.0.1:8000/app/](http://127.0.0.1:8000/app/)**

*(You can also double-click `frontend/index.html` to run as a local file; it will connect to the backend through CORS automatically).*

---



