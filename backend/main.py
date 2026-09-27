"""
backend/main.py
===============
FastAPI Backend for Bioactive Molecule Predictor

Endpoints:
  GET  /          -> API information and active model
  GET  /health    -> Health check status
  POST /predict   -> Predict bioactivity from a molecular SMILES string
  GET  /app       -> Serves the frontend web interface directly

The backend loads the trained best model (saved during train_model.py)
and performs real-time inference using RDKit feature generation.
"""

import os
import json
import numpy as np
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")   # Suppress chemistry parsing warnings

# --- Absolute path resolution (runs from any directory) ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
MODEL_PATH = os.path.join(CURRENT_DIR, "model", "best_model.pkl")
METADATA_PATH = os.path.join(CURRENT_DIR, "model", "metadata.json")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

# --- Load Model & Metadata at Startup ---
try:
    best_model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    BEST_MODEL_NAME = metadata["best_model_name"]
    REVERSE_MAPPING = {int(k): v for k, v in metadata["reverse_mapping"].items()}
    FINGERPRINT_RADIUS = int(metadata.get("fingerprint_radius", 2))
    FINGERPRINT_BITS = int(metadata.get("fingerprint_bits", 2048))
    print(f"[OK] Model successfully loaded: {BEST_MODEL_NAME}")
except Exception as e:
    raise RuntimeError(
        f"Failed to load model from {MODEL_PATH}. Run train_model.py first.\nError: {e}"
    )

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Bioactive Molecule Predictor",
    description="REST API predicting molecular bioactivity with Machine Learning.",
    version="1.0.0"
)

# --- CORS Middleware ---
# Allows browsers opening frontend locally (via file:// or http://) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request Validation Schema ---
class PredictRequest(BaseModel):
    smiles: str


# --- Feature Extraction Helper ---
def smiles_to_fingerprint(smiles: str):
    """
    Converts a user-provided SMILES string into a 2048-bit Morgan fingerprint vector.
    Matches the exact feature generation logic used in training.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol, FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS
    )
    return np.array(fp, dtype=np.uint8).reshape(1, -1)


# --- API Routes ---
@app.get("/")
def root():
    """API welcome message with current model information."""
    return {
        "message": "Bioactive Molecule Predictor API",
        "model": BEST_MODEL_NAME,
        "docs": "/docs",
        "app": "/app/"
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": BEST_MODEL_NAME
    }


@app.post("/predict")
def predict(request: PredictRequest):
    """
    Prediction Pipeline:
      1. Validate non-empty SMILES string
      2. Convert SMILES to 2048-bit Morgan Fingerprint via RDKit
      3. Feed features into saved best model
      4. Calculate classification and confidence probability
      5. Return dynamic JSON response
    """
    clean_smiles = request.smiles.strip()
    if not clean_smiles:
        raise HTTPException(status_code=422, detail="SMILES string cannot be empty.")

    # Convert to chemical features
    features = smiles_to_fingerprint(clean_smiles)
    if features is None:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid SMILES string '{clean_smiles}'. Chemistry parser could not resolve structure."
        )

    # Predict using loaded model
    try:
        pred_int = int(best_model.predict(features)[0])
        pred_label = REVERSE_MAPPING.get(pred_int, "active" if pred_int == 1 else "inactive")

        # Probability score
        if hasattr(best_model, "predict_proba"):
            probs = best_model.predict_proba(features)[0]
            confidence = float(probs[pred_int])
        elif hasattr(best_model, "decision_function"):
            score = float(best_model.decision_function(features)[0])
            confidence = float(1.0 / (1.0 + np.exp(-score)))
            if pred_int == 0:
                confidence = 1.0 - confidence
        else:
            confidence = 1.0

    except Exception as err:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {str(err)}"
        )

    return {
        "prediction": pred_label,
        "probability": round(confidence, 4),
        "model": BEST_MODEL_NAME,
        "smiles": clean_smiles
    }


# --- Mount Frontend static app at /app ---
if os.path.exists(FRONTEND_DIR):
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
