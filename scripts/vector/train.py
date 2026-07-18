"""
EEG-to-Text Embedding Model
============================
Data format (train.json / test.json):
  List of samples, each with:
    input:  list of 10 x {
                time: float,
                eeg:  {delta, theta, loAlpha, hiAlpha, loBeta, hiBeta, loGamma, midGamma}
                      (raw power-spectral-density integers, order of magnitude ~1e6–1e7)
                      OR null (handled with zeros)
            }
    output: {text: str, time: [start, end], dimensions: 384, embedding: [384 floats]}

Task: Given a 10-step EEG band-power sequence (~103 ms/step), predict the
      384-dimensional sentence embedding of the text being read.

Architecture: BiLSTM encoder → LayerNorm → Linear head (→ 384 dims)
Loss:         MSE + cosine-similarity loss (balanced 50/50)

EEG normalisation:
  Step 1 – log1p: compresses million-scale PSD values to ~[14, 16]
  Step 2 – per-channel z-score over the training set: zero-mean, unit-variance
"""

import json
import math
import os
import sys
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# Canonical EEG band order (low → high frequency)
EEG_BANDS = ["delta", "theta", "loAlpha", "hiAlpha", "loBeta", "hiBeta", "loGamma", "midGamma"]


# ─── 1. Dataset ──────────────────────────────────────────────────────────────

class EEGDataset(Dataset):
    """
    Load EEG band-power sequences and target sentence embeddings from a JSON file.

    EEG normalisation pipeline (fitted on THIS dataset):
      1. log1p  – compresses million-scale PSD integers to a tighter range
      2. z-score – per-band mean/std computed from all timesteps in this file
    """

    def __init__(self, json_path: str, norm_stats: dict | None = None):
        """
        Args:
            json_path:   Path to train.json or test.json.
            norm_stats:  If provided (dict with keys 'time_mean', 'time_std',
                         'eeg_mean', 'eeg_std'), use these instead of computing
                         from this file.  Pass the training stats when loading
                         test data so both sets are on the same scale.
        """
        with open(json_path) as f:
            raw = json.load(f)

        self.samples    = raw
        self.eeg_bands  = EEG_BANDS
        self.eeg_channels = len(EEG_BANDS)            # always 8
        self.feature_dim  = 1 + self.eeg_channels     # time + 8 bands = 9

        # ── Fit normalisation stats (or inherit from training set) ──────────
        if norm_stats is not None:
            self.time_mean = norm_stats["time_mean"]
            self.time_std  = norm_stats["time_std"]
            self.eeg_mean  = norm_stats["eeg_mean"]   # list[8]
            self.eeg_std   = norm_stats["eeg_std"]    # list[8]
        else:
            self._fit_norm_stats(raw)

        has_eeg = self._has_eeg(raw)
        print(f"[Dataset] {json_path}: {len(raw)} samples | "
              f"seq_len=10 | EEG={'8 bands' if has_eeg else 'null→zeros'} | "
              f"feature_dim={self.feature_dim}")

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _has_eeg(raw) -> bool:
        for s in raw:
            for item in s["input"]:
                if isinstance(item.get("eeg"), dict):
                    return True
        return False

    def _fit_norm_stats(self, raw):
        """Compute time z-score params and per-band log1p z-score params."""
        all_times = [item["time"] for s in raw for item in s["input"]]
        self.time_mean = sum(all_times) / len(all_times)
        self.time_std  = math.sqrt(
            sum((t - self.time_mean) ** 2 for t in all_times) / len(all_times)
        ) or 1.0

        # Collect log1p band values per band
        band_vals = {b: [] for b in EEG_BANDS}
        for s in raw:
            for item in s["input"]:
                eeg = item.get("eeg")
                if isinstance(eeg, dict):
                    for b in EEG_BANDS:
                        v = eeg.get(b, 0.0) or 0.0
                        band_vals[b].append(math.log1p(float(v)))

        # Per-band mean & std (fallback to 0/1 if no EEG data)
        self.eeg_mean = []
        self.eeg_std  = []
        for b in EEG_BANDS:
            vals = band_vals[b]
            if vals:
                mu  = sum(vals) / len(vals)
                std = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals)) or 1.0
            else:
                mu, std = 0.0, 1.0
            self.eeg_mean.append(mu)
            self.eeg_std.append(std)

    def norm_stats(self) -> dict:
        """Return serialisable stats dict — pass to test EEGDataset."""
        return {
            "time_mean": self.time_mean,
            "time_std":  self.time_std,
            "eeg_mean":  self.eeg_mean,
            "eeg_std":   self.eeg_std,
        }

    def _extract_eeg(self, eeg_field) -> list:
        """
        Extract 8 normalised band values from an EEG dict (or return zeros).
        Pipeline: raw → log1p → z-score
        """
        if not isinstance(eeg_field, dict):
            return [0.0] * self.eeg_channels

        vals = []
        for i, b in enumerate(EEG_BANDS):
            raw_val = eeg_field.get(b, 0.0) or 0.0
            log_val = math.log1p(float(raw_val))
            normed  = (log_val - self.eeg_mean[i]) / self.eeg_std[i]
            vals.append(normed)
        return vals

    # ── PyTorch interface ─────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        rows = []
        for item in sample["input"]:
            t_norm  = (item["time"] - self.time_mean) / self.time_std
            eeg_vals = self._extract_eeg(item.get("eeg"))
            rows.append([t_norm] + eeg_vals)

        x    = torch.tensor(rows, dtype=torch.float32)                          # [10, 9]
        y    = torch.tensor(sample["output"]["embedding"], dtype=torch.float32)  # [384]
        text = sample["output"]["text"]
        return x, y, text


# ─── 2. Model ─────────────────────────────────────────────────────────────────

class EEGToEmbedding(nn.Module):
    """
    Bidirectional LSTM encoder that maps a variable-length EEG sequence
    to a 384-dimensional sentence embedding.

    Input:  [batch, seq_len, feature_dim]
    Output: [batch, embed_dim]
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        embed_dim: int = 384,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        # x: [B, seq_len, feature_dim]
        x = self.input_proj(x)                  # [B, seq_len, hidden_dim]
        out, _ = self.lstm(x)                   # [B, seq_len, 2*hidden_dim]
        # mean-pool over time
        pooled = out.mean(dim=1)                # [B, 2*hidden_dim]
        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)
        return self.head(pooled)                # [B, 384]


# ─── 3. Loss ──────────────────────────────────────────────────────────────────

def combined_loss(pred: torch.Tensor, target: torch.Tensor,
                  alpha: float = 0.5) -> torch.Tensor:
    """MSE + (1 - cosine similarity), weighted by alpha."""
    mse = F.mse_loss(pred, target)
    cos = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
    return alpha * mse + (1 - alpha) * cos


# ─── 4. Training ──────────────────────────────────────────────────────────────

def train(
    train_path: str = "train.json",
    test_path: str = "test.json",
    epochs: int = 200,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    batch_size: int = 4,
    save_path: str = "eeg_model.pt",
    seed: int = 42,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Train] Device: {device}")

    # ── Datasets & loaders ──────────────────────────────────────────────────
    train_ds = EEGDataset(train_path)
    train_loader = DataLoader(
        train_ds, batch_size=min(batch_size, len(train_ds)),
        shuffle=True, drop_last=False,
        collate_fn=lambda b: (
            torch.stack([i[0] for i in b]),
            torch.stack([i[1] for i in b]),
            [i[2] for i in b],
        ),
    )

    has_test = os.path.exists(test_path)
    if has_test:
        # ⚠️  Always use TRAINING norm stats for test data — prevents data leakage
        test_ds = EEGDataset(test_path, norm_stats=train_ds.norm_stats())
        test_loader = DataLoader(
            test_ds, batch_size=min(batch_size, len(test_ds)),
            shuffle=False, drop_last=False,
            collate_fn=lambda b: (
                torch.stack([i[0] for i in b]),
                torch.stack([i[1] for i in b]),
                [i[2] for i in b],
            ),
        )
    else:
        print(f"[Train] Warning: {test_path} not found – skipping evaluation.")
        test_loader = None

    # ── Model, optimiser, scheduler ─────────────────────────────────────────
    model = EEGToEmbedding(
        feature_dim=train_ds.feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        embed_dim=train_ds.samples[0]["output"]["dimensions"],
        dropout=dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Training loop ────────────────────────────────────────────────────────
    best_loss = float("inf")
    log_every = max(1, epochs // 10)

    print(f"\n[Train] Starting training for {epochs} epochs…\n")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = combined_loss(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "feature_dim": train_ds.feature_dim,
                    "embed_dim": train_ds.samples[0]["output"]["dimensions"],
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    # Normalisation stats (needed for inference)
                    "norm_stats": train_ds.norm_stats(),
                    "eeg_bands": train_ds.eeg_bands,
                    "eeg_channels": train_ds.eeg_channels,
                    # Store all training texts + embeddings for retrieval
                    "train_texts": [s["output"]["text"] for s in train_ds.samples],
                    "train_embeddings": [s["output"]["embedding"] for s in train_ds.samples],
                },
                save_path,
            )

        if epoch % log_every == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4}/{epochs}  loss={avg_loss:.6f}  "
                  f"best={best_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

    print(f"\n[Train] Best training loss: {best_loss:.6f}  →  saved to '{save_path}'")

    # ── Evaluation on test set ───────────────────────────────────────────────
    if test_loader is not None:
        evaluate(model, test_loader, train_ds, device)


# ─── 5. Evaluation ────────────────────────────────────────────────────────────

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.unsqueeze(0), b, dim=-1)


def evaluate(model, test_loader, train_ds, device):
    """
    For each test sample:
      1. Compute the predicted embedding.
      2. Find the closest text in the TRAINING set by cosine similarity.
      3. Report cosine similarity vs. ground-truth embedding.
    """
    model.eval()

    # Pre-compute training embeddings tensor for retrieval
    train_embs = torch.tensor(
        [s["output"]["embedding"] for s in train_ds.samples],
        dtype=torch.float32,
    ).to(device)  # [N_train, 384]

    train_texts = [s["output"]["text"] for s in train_ds.samples]

    total_cos = 0.0
    total_mse = 0.0
    n = 0

    print("\n" + "=" * 72)
    print("EVALUATION ON TEST SET")
    print("=" * 72)

    with torch.no_grad():
        for x, y, texts in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)                              # [B, 384]

            for i in range(len(texts)):
                pred_i = preds[i]                         # [384]
                gt_i   = y[i]                             # [384]

                # Similarity with ground-truth
                cos_gt = F.cosine_similarity(pred_i.unsqueeze(0), gt_i.unsqueeze(0)).item()
                mse    = F.mse_loss(pred_i, gt_i).item()

                # Retrieve closest training sample
                sims   = cosine_sim(pred_i, train_embs)   # [N_train]
                best_idx = sims.argmax().item()
                retrieved_text = train_texts[best_idx]
                retrieved_sim  = sims[best_idx].item()

                total_cos += cos_gt
                total_mse += mse
                n += 1

                print(f"\n  Ground truth:  \"{texts[i]}\"")
                print(f"  Retrieved:     \"{retrieved_text}\"  (cos={retrieved_sim:.4f})")
                print(f"  GT cos sim:    {cos_gt:.4f}   MSE: {mse:.6f}")

    print("\n" + "-" * 72)
    print(f"  Mean cosine similarity (pred vs GT):  {total_cos/n:.4f}")
    print(f"  Mean MSE       (pred vs GT):          {total_mse/n:.6f}")
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
    
    train(
        train_path = train_file,
        test_path = test_file,
        save_path = save_file
    )

if __name__ == "__main__":
    main()