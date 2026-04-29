import torch, joblib, os
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from collections import Counter
import math

MODEL_PATH = "/home/aparna/ml_scripts/ae_model.pth"
ISO_PATH = "/home/aparna/ml_scripts/iso_forest_model.pkl"
SCALER_PATH = "/home/aparna/ml_scripts/scaler.pkl"
LOG_FILE = "/home/aparna/ml_log.txt"
LOCK_FILE = "/tmp/ml_lock"
TARGET_FILE = "/home/aparna/protected/test.txt"

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
    return np.array([len(path), stat.st_size, 1.0, get_entropy(path), stat.st_uid])

def log_event(msg):
    with open(LOG_FILE, "a") as f: f.write(msg + "\n")

try:
    open(LOCK_FILE, 'w').close()
    features = extract_features(TARGET_FILE).reshape(1, -1)
    base_data = np.repeat(features, 10, axis=0)
    
    scaler = MinMaxScaler()
    scaled_data = torch.FloatTensor(scaler.fit_transform(base_data))
    
    ae = Autoencoder()
    opt = torch.optim.Adam(ae.parameters(), lr=0.01)
    for _ in range(50):
        opt.zero_grad()
        loss = nn.MSELoss()(ae(scaled_data), scaled_data)
        loss.backward()
        opt.step()
    
    iso = IsolationForest(contamination=0.1).fit(base_data)
    
    torch.save(ae.state_dict(), MODEL_PATH)
    joblib.dump(iso, ISO_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    os.system("git -C /home/aparna/protected add test.txt")
    os.system('git -C /home/aparna/protected commit -m "Auto-baseline update"')
    log_event("✅ Models updated and Baseline committed.")

except Exception as e:
    log_event(f"❌ Retraining Error: {str(e)}")
finally:
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)
