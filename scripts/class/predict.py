import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import math
import sys
import time
import json
import os
from time import sleep
import requests
import numpy as np
from numpy.linalg import norm

EEG_BANDS = ["delta", "theta", "loAlpha", "hiAlpha", "loBeta", "hiBeta", "loGamma", "midGamma"]
BASE_URL = "http://localhost:3000"
END_POINT = f"{BASE_URL}/mindwave/data"

class LabelEncoder:
    """
    Bidirectional mapping between text strings and integer class indices.
    Always built from the training set; test labels must be a subset.
    """
 
    def __init__(self, texts: list[str]):
        # Preserve insertion order so index is deterministic
        seen = []
        for t in texts:
            if t not in seen:
                seen.append(t)
        self.classes_   = seen                          # list[str]
        self.num_classes = len(seen)
        self._str2idx   = {t: i for i, t in enumerate(seen)}
 
    def encode(self, text: str) -> int:
        if text not in self._str2idx:
            raise ValueError(
                f"Label '{text}' was not seen during training.\n"
                f"Known labels: {self.classes_}"
            )
        return self._str2idx[text]
 
    def decode(self, idx: int) -> str:
        return self.classes_[idx]
 
    def __len__(self):
        return self.num_classes
 
 
# ─── 2. Dataset ───────────────────────────────────────────────────────────────
 
class EEGDataset(Dataset):
    """
    Loads EEG sequences and text labels from a JSON file.
 
    Each sample returns:
        x     – FloatTensor [10, 9]  (normalised time + 8 EEG bands)
        label – LongTensor  []       (class index)
        text  – str                  (original label string)
 
    EEG normalisation (must use TRAINING stats for both train & test):
        1. log1p  – compresses million-scale PSD integers
        2. z-score – per-band, fitted on the training set
    """
 
    def __init__(
        self,
        json_path: str,
        label_encoder: LabelEncoder | None = None,
        norm_stats: dict | None = None,
    ):
        """
        Args:
            json_path:     Path to train.json or test.json.
            label_encoder: Pass the training LabelEncoder when loading test
                           data so label indices are consistent.
            norm_stats:    Pass training norm stats when loading test data.
        """
        with open(json_path) as f:
            raw = json.load(f)
        self.samples = raw
 
        # ── Label encoder ───────────────────────────────────────────────────
        if label_encoder is not None:
            self.label_encoder = label_encoder
        else:
            texts = [s["output"]["text"] for s in raw]
            self.label_encoder = LabelEncoder(texts)
 
        # ── Normalisation stats ─────────────────────────────────────────────
        if norm_stats is not None:
            self.time_mean = norm_stats["time_mean"]
            self.time_std  = norm_stats["time_std"]
            self.eeg_mean  = norm_stats["eeg_mean"]
            self.eeg_std   = norm_stats["eeg_std"]
        else:
            self._fit_norm_stats(raw)
 
        has_eeg = any(
            isinstance(item.get("eeg"), dict)
            for s in raw for item in s["input"]
        )
        print(
            f"[Dataset] {json_path}: {len(raw)} samples | "
            f"classes={self.label_encoder.num_classes} | "
            f"EEG={'8 bands' if has_eeg else 'null→zeros'}"
        )
 
    # ── Normalisation fitting ─────────────────────────────────────────────────
 
    def _fit_norm_stats(self, raw):
        all_times = [item["time"] for s in raw for item in s["input"]]
        self.time_mean = sum(all_times) / len(all_times)
        self.time_std  = math.sqrt(
            sum((t - self.time_mean) ** 2 for t in all_times) / len(all_times)
        ) or 1.0
 
        band_log_vals = {b: [] for b in EEG_BANDS}
        for s in raw:
            for item in s["input"]:
                eeg = item.get("eeg")
                if isinstance(eeg, dict):
                    for b in EEG_BANDS:
                        v = eeg.get(b, 0.0) or 0.0
                        band_log_vals[b].append(math.log1p(float(v)))
 
        self.eeg_mean, self.eeg_std = [], []
        for b in EEG_BANDS:
            vals = band_log_vals[b]
            if vals:
                mu  = sum(vals) / len(vals)
                std = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals)) or 1.0
            else:
                mu, std = 0.0, 1.0
            self.eeg_mean.append(mu)
            self.eeg_std.append(std)
 
    def norm_stats(self) -> dict:
        return {
            "time_mean": self.time_mean,
            "time_std":  self.time_std,
            "eeg_mean":  self.eeg_mean,
            "eeg_std":   self.eeg_std,
        }
 
    # ── Feature extraction ────────────────────────────────────────────────────
 
    def _extract_eeg(self, eeg_field) -> list[float]:
        if not isinstance(eeg_field, dict):
            return [0.0] * len(EEG_BANDS)
        vals = []
        for i, b in enumerate(EEG_BANDS):
            raw_v  = eeg_field.get(b, 0.0) or 0.0
            log_v  = math.log1p(float(raw_v))
            normed = (log_v - self.eeg_mean[i]) / self.eeg_std[i]
            vals.append(normed)
        return vals
 
    # ── PyTorch interface ─────────────────────────────────────────────────────
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text   = sample["output"]["text"]
 
        rows = []
        for item in sample["input"]:
            t_norm   = (item["time"] - self.time_mean) / self.time_std
            eeg_vals = self._extract_eeg(item.get("eeg"))
            rows.append([t_norm] + eeg_vals)
 
        x     = torch.tensor(rows, dtype=torch.float32)                    # [10, 9]
        label = torch.tensor(self.label_encoder.encode(text), dtype=torch.long)
        return x, label, text
 
 
# ─── 3. Collate ───────────────────────────────────────────────────────────────
 
def collate_fn(batch):
    xs, labels, texts = zip(*batch)
    return torch.stack(xs), torch.stack(labels), list(texts)
 
 
# ─── 4. Model ─────────────────────────────────────────────────────────────────
 
class EEGClassifier(nn.Module):
    """
    BiLSTM encoder → mean pool → LayerNorm → classification head.
 
    Input:  [batch, seq_len, feature_dim]   (feature_dim = 9)
    Output: [batch, num_classes]            (raw logits, no softmax)
    """
 
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float  = 0.4,
    ):
        super().__init__()
 
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
 
        self.lstm = nn.LSTM(
            input_size  = hidden_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            bidirectional = True,
            dropout = dropout if num_layers > 1 else 0.0,
        )
 
        self.norm    = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
 
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),   # logits over N text labels
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x       = self.input_proj(x)          # [B, 10, hidden]
        out, _  = self.lstm(x)                # [B, 10, 2*hidden]
        pooled  = out.mean(dim=1)             # [B, 2*hidden]
        pooled  = self.norm(pooled)
        pooled  = self.dropout(pooled)
        return self.head(pooled)              # [B, num_classes]

def load_model(save_path: str = "eeg_classifier.pt"):
    """
    Load a saved classifier checkpoint.
 
    Returns:
        model         – EEGClassifier in eval mode
        label_encoder – LabelEncoder with the training label set
        ckpt          – full checkpoint dict (includes norm_stats, eeg_bands)
 
    Example:
        model, le, ckpt = load_model("eeg_classifier.pt")
        text, confidence = predict(model, le, sequence, ckpt["norm_stats"])
        print(f"Predicted: '{text}'  ({confidence:.1%})")
    """
    ckpt = torch.load(save_path, map_location="cpu")
    le   = LabelEncoder(ckpt["classes"])
    model = EEGClassifier(
        feature_dim = 9,
        num_classes = ckpt["num_classes"],
        hidden_dim  = ckpt["hidden_dim"],
        num_layers  = ckpt["num_layers"],
        dropout     = ckpt["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, le, ckpt
 
 
def predict(
    model:         EEGClassifier,
    label_encoder: LabelEncoder,
    eeg_sequence:  list,
    norm_stats:    dict,
    top_k:         int = 3,
) -> tuple[str, float, list[tuple[str, float]]]:
    """
    Predict the text label for a single EEG sequence.
 
    Args:
        model:        Loaded EEGClassifier.
        label_encoder: From load_model().
        eeg_sequence: List of {"eeg": {band: val, ...}, "time": float}
        norm_stats:   ckpt["norm_stats"] from load_model().
        top_k:        How many ranked candidates to return.
 
    Returns:
        (best_text, best_confidence, [(text, confidence), ...] top_k list)
 
    Example:
        model, le, ckpt = load_model()
        text, conf, ranked = predict(model, le, sequence, ckpt["norm_stats"])
        print(f"Predicted: '{text}'  ({conf:.1%})")
    """
    time_mean = norm_stats["time_mean"]
    time_std  = norm_stats["time_std"]
    eeg_mean  = norm_stats["eeg_mean"]
    eeg_std   = norm_stats["eeg_std"]
 
    rows = []
    for item in eeg_sequence:
        t_norm = (item["time"] - time_mean) / time_std
        eeg    = item.get("eeg")
        if isinstance(eeg, dict):
            eeg_vals = []
            for i, b in enumerate(EEG_BANDS):
                raw_v  = eeg.get(b, 0.0) or 0.0
                log_v  = math.log1p(float(raw_v))
                normed = (log_v - eeg_mean[i]) / eeg_std[i]
                eeg_vals.append(normed)
        else:
            eeg_vals = [0.0] * len(EEG_BANDS)
        rows.append([t_norm] + eeg_vals)
 
    x = torch.tensor([rows], dtype=torch.float32)  # [1, 10, 9]
    with torch.no_grad():
        logits = model(x)[0]                        # [num_classes]
        probs  = F.softmax(logits, dim=-1)
 
    top_probs, top_idxs = probs.topk(min(top_k, len(label_encoder)))
    ranked = [
        (label_encoder.decode(idx.item()), prob.item())
        for prob, idx in zip(top_probs, top_idxs)
    ]
    return ranked[0][0], ranked[0][1], ranked
 

def grab_eeg_data():
    response  = requests.get(END_POINT).json()
    return response

def main(arguments = []):
    delay = float(arguments[0]) if len(arguments) > 0 else .1
    model, le, ckpt = load_model(save_path = "output/modelClass.pt")
    
    input_dir = "input"
    test_dir = os.path.join(input_dir, "test")
    test_file = os.path.join(test_dir, "test.json")

    with open(test_file, "r") as f:
        samples = json.load(f)
    
    count = 0
    total = len(samples)
    for sample in samples:
        input_packet = sample["input"]
        output_data = sample["output"]

        text, confidence, ranked = predict(model, le, input_packet, ckpt["norm_stats"])

        if text == output_data["text"]:
            count += 1
    
    print(f"{count=} {total=} {count/total=}")
    
        

if __name__ == "__main__":
    main(sys.argv[1:])