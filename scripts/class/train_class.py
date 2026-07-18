"""
EEG Text Classifier
====================
Task:   Given a 10-step sequence of EEG band-power readings, predict which
        text string (from the training label set) was being read.
 
Input:  {"eeg": {delta, theta, loAlpha, hiAlpha, loBeta, hiBeta,
                  loGamma, midGamma}, "time": float}  × 10 steps
Output: One of N text labels defined by the training data.
 
Architecture:
    Input (9 features: 8 EEG bands + time)
      → Linear projection
      → Bidirectional LSTM
      → Mean pool over time
      → LayerNorm + Dropout
      → Linear head (→ N classes)
 
Loss: CrossEntropyLoss with label smoothing
      (smoothing prevents overconfidence on tiny datasets)
 
No embeddings are used anywhere in this file.
"""
 
import json
import math
import os
import random
import sys
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
 
 
# ─── Constants ────────────────────────────────────────────────────────────────
 
EEG_BANDS = ["delta", "theta", "loAlpha", "hiAlpha",
             "loBeta", "hiBeta", "loGamma", "midGamma"]
 
 
# ─── 1. Label encoder ─────────────────────────────────────────────────────────
 
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
 
 
# ─── 5. Training ──────────────────────────────────────────────────────────────
 
def train(
    train_path:  str   = "train.json",
    test_path:   str   = "test.json",
    epochs:      int   = 200,
    lr:          float = 1e-3,
    hidden_dim:  int   = 128,
    num_layers:  int   = 2,
    dropout:     float = 0.3,
    batch_size:  int   = 4,
    label_smooth: float = 0.1,
    save_path:   str   = "eeg_classifier.pt",
    seed:        int   = 42,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Train] Device: {device}")
 
    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = EEGDataset(train_path)
    train_loader = DataLoader(
        train_ds,
        batch_size = min(batch_size, len(train_ds)),
        shuffle    = True,
        drop_last  = False,
        collate_fn = collate_fn,
    )
 
    has_test = os.path.exists(test_path)
    if has_test:
        test_ds = EEGDataset(
            test_path,
            label_encoder = train_ds.label_encoder,  # same label→index mapping
            norm_stats    = train_ds.norm_stats(),    # same EEG normalisation
        )
        test_loader = DataLoader(
            test_ds,
            batch_size = min(batch_size, len(test_ds)),
            shuffle    = False,
            collate_fn = collate_fn,
        )
    else:
        print(f"[Train] {test_path} not found – skipping evaluation.")
        test_loader = None
 
    # ── Model ────────────────────────────────────────────────────────────────
    num_classes = train_ds.label_encoder.num_classes
    model = EEGClassifier(
        feature_dim = 9,              # time + 8 EEG bands
        num_classes = num_classes,
        hidden_dim  = hidden_dim,
        num_layers  = num_layers,
        dropout     = dropout,
    ).to(device)
 
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Model params: {total_params:,} | Classes: {num_classes}")
 
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
 
    # ── Loop ─────────────────────────────────────────────────────────────────
    best_loss = float("inf")
    log_every = max(1, epochs // 10)
 
    print(f"\n[Train] {epochs} epochs | label_smoothing={label_smooth}\n")
 
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss  = 0.0
        total_correct = 0
        total_n       = 0
 
        for x, labels, _ in train_loader:
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(x)                          # [B, C]
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
 
            total_loss    += loss.item()
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_n       += len(labels)
 
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_n * 100
 
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch":         epoch,
                    "model_state":   model.state_dict(),
                    "hidden_dim":    hidden_dim,
                    "num_layers":    num_layers,
                    "dropout":       dropout,
                    "num_classes":   num_classes,
                    "classes":       train_ds.label_encoder.classes_,
                    "norm_stats":    train_ds.norm_stats(),
                    "eeg_bands":     EEG_BANDS,
                },
                save_path,
            )
 
        if epoch % log_every == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:>4}/{epochs} | "
                f"loss={avg_loss:.4f}  best={best_loss:.4f} | "
                f"train_acc={train_acc:.1f}% | "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )
 
    print(f"\n[Train] Best loss: {best_loss:.4f}  →  saved to '{save_path}'")
 
    if test_loader is not None:
        # Reload best checkpoint for evaluation
        best_ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state"])
        evaluate(model, test_loader, train_ds.label_encoder, device)
 
 
# ─── 6. Evaluation ────────────────────────────────────────────────────────────
 
def evaluate(
    model:         nn.Module,
    test_loader:   DataLoader,
    label_encoder: LabelEncoder,
    device:        torch.device,
):
    """
    Run the best saved model on the test set.
    Reports per-sample predictions and overall top-1 / top-3 accuracy.
    """
    model.eval()
 
    all_preds   = []
    all_labels  = []
    all_texts   = []
    all_logits  = []
 
    with torch.no_grad():
        for x, labels, texts in test_loader:
            x, labels = x.to(device), labels.to(device)
            logits = model(x)                   # [B, C]
            all_logits.append(logits.cpu())
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_texts.extend(texts)
 
    logits_all = torch.cat(all_logits, dim=0)  # [N, C]
    probs_all  = F.softmax(logits_all, dim=-1) # [N, C]
 
    # ── Per-sample results ───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("EVALUATION ON TEST SET")
    print("=" * 72)
 
    n = len(all_texts)
    top1_correct = 0
    top3_correct = 0
    top_k = min(3, label_encoder.num_classes)
 
    for i in range(n):
        gt_text   = all_texts[i]
        gt_idx    = all_labels[i]
        pred_idx  = all_preds[i]
        pred_text = label_encoder.decode(pred_idx)
        correct   = pred_idx == gt_idx
 
        # Top-k predictions with confidence
        top_probs, top_idxs = probs_all[i].topk(top_k)
        top_predictions = [
            f"{label_encoder.decode(idx.item())!r} ({prob:.1%})"
            for prob, idx in zip(top_probs, top_idxs)
        ]
 
        top1_correct += int(correct)
        top3_correct += int(gt_idx in top_idxs.tolist())
 
        status = "✓" if correct else "✗"
        print(f"\n  {status} Ground truth:  \"{gt_text}\"")
        print(f"    Predicted:     \"{pred_text}\"  "
              f"({'CORRECT' if correct else 'WRONG'})")
        print(f"    Top-{top_k} candidates:")
        for rank, p in enumerate(top_predictions, 1):
            marker = " ←" if rank == 1 else ""
            print(f"      {rank}. {p}{marker}")
 
    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "-" * 72)
    print(f"  Top-1 accuracy:  {top1_correct}/{n} = {top1_correct/n:.1%}")
    if top_k > 1:
        print(f"  Top-{top_k} accuracy:  {top3_correct}/{n} = {top3_correct/n:.1%}")
    print("=" * 72 + "\n")

def main():
    input_dir = "input"
    
    train_dir = os.path.join(input_dir, "train")
    if not os.path.exists(train_dir):
        sys.exit(1)
    
    test_dir = os.path.join(input_dir, "test")
    if not os.path.exists(test_dir):
        sys.exit(1)

    train_file = os.path.join(train_dir, "train.json")
    test_file = os.path.join(test_dir, "test.json")
    
    save_dir = "output"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = os.path.join(save_dir, "modelVector.pt")
    
    save_dir = "output"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = os.path.join(save_dir, "modelClass.pt")
    
    train(
        train_path = train_file,
        test_path = test_file,
        save_path = save_file
    )

if __name__ == "__main__":
    main()