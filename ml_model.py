import sys, os, json, torch, joblib, math, subprocess
import torch.nn as nn
import numpy as np
from collections import Counter

# --- CONFIGURATION ---
MODEL_PATH = "/home/aparna/ml_scripts/ae_model.pth"
ISO_PATH = "/home/aparna/ml_scripts/iso_forest_model.pkl"
SCALER_PATH = "/home/aparna/ml_scripts/scaler.pkl"
LOG_FILE = "/home/aparna/ml_log.txt"
LOCK_FILE = "/tmp/ml_lock"

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(5, 3), nn.ReLU(), nn.Linear(3, 2))
        self.decoder = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 5))
    def forward(self, x): return self.decoder(self.encoder(x))

def get_entropy(path):
    with open(path, 'rb') as f: data = f.read()
    if not data: return 0
    counts = Counter(data)
    return -sum((c/len(data)) * math.log2(c/len(data)) for c in counts.values())

def extract_features(path):
    stat = os.stat(path)
    # path_len, file_size, constant, entropy, uid
    return np.array([len(path), stat.st_size, 1.0, get_entropy(path), stat.st_uid])

def log_event(msg):
    with open(LOG_FILE, "a") as f: f.write(msg + "\n")

try:
    with open(sys.argv[1], 'r') as f: alert = json.load(f)
    file_path = alert['parameters']['alert']['syscheck']['path']
    
    features = extract_features(file_path).reshape(1, -1)
    scaler = joblib.load(SCALER_PATH)
    features_scaled = torch.FloatTensor(scaler.transform(features))
    
    ae = Autoencoder()
    ae.load_state_dict(torch.load(MODEL_PATH))
    iso = joblib.load(ISO_PATH)
    
    ae.eval()
    with torch.no_grad():
        loss = nn.MSELoss()(ae(features_scaled), features_scaled).item()
    iso_pred = iso.predict(features)[0]

    # --- CALIBRATED LOGIC ---
    is_malicious = False
    if loss > 1000: is_malicious = True
    elif loss > 50 and iso_pred == -1: is_malicious = True
    elif features[0][3] > 6.5: is_malicious = True

    status = "🚨 MALICIOUS" if is_malicious else "✅ AUTHORIZED"
    
    log_event(f"--- [ {status} ] ---")
    log_event(f"Target: {file_path}\nMetrics: Loss: {loss:.4f} | Ent: {features[0][3]:.2f}")

    if is_malicious:
        os.system(f"git -C /home/aparna/protected reset --hard HEAD")
        log_event("🚨 Unauthorized: Restoring repo")
    else:
        if os.path.exists(LOCK_FILE):
            log_event("⏳ Retraining already in progress. Skipping...")
        else:
            log_event("🔄 Starting Adaptive Retraining...")
            subprocess.Popen(["/home/aparna/ml_venv/bin/python3", "/home/aparna/ml_scripts/retrain.py"])

except Exception as e:
    log_event(f"Error in ML Model: {str(e)}")
