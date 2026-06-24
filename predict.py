import torch
import torch
import torch.nn as nn
import math
import sys
import time
from time import sleep
import requests

EEG_BANDS = ["delta", "theta", "loAlpha", "hiAlpha", "loBeta", "hiBeta", "loGamma", "midGamma"]
BASE_URL = "http://localhost:3000"
END_POINT = f"{BASE_URL}/mindwave/data"

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


def load_model(save_path: str = "eeg_model.pt"):
    """Load a saved checkpoint and return (model, checkpoint_dict)."""
    ckpt = torch.load(save_path, map_location="cpu")
    model = EEGToEmbedding(
        feature_dim=ckpt["feature_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_layers=ckpt["num_layers"],
        embed_dim=ckpt["embed_dim"],
        dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


 
def predict(model, eeg_sequence: list, norm_stats: dict,
            eeg_bands: list = EEG_BANDS) -> torch.Tensor:
    """
    Run inference on a single EEG window captured in real time.
 
    Args:
        model:        Loaded EEGToEmbedding model (eval mode).
        eeg_sequence: List of 10 x {"eeg": {...dict or None...}, "time": float}
                      Timestamps can be anything (session clock, epoch time, etc.)
                      — they are made window-relative before normalisation, so the
                      model never sees the raw session clock value.
        norm_stats:   Dict from checkpoint["norm_stats"] with keys
                      time_mean, time_std, eeg_mean (list[8]), eeg_std (list[8]).
        eeg_bands:    Band name order (default: EEG_BANDS).
 
    Returns:
        torch.Tensor of shape [384] — predicted sentence embedding.
 
    Example:
        model, ckpt = load_model("eeg_model.pt")
        embedding   = predict(model, my_sequence, ckpt["norm_stats"])
    """
    time_mean = norm_stats["time_mean"]
    time_std  = norm_stats["time_std"]
    eeg_mean  = norm_stats["eeg_mean"]
    eeg_std   = norm_stats["eeg_std"]
 
    # Anchor to the window's own first timestamp — safe at any session length
    t0 = eeg_sequence[0]["time"]
 
    rows = []
    for item in eeg_sequence:
        t_rel  = item["time"] - t0             # always [0, ~0.93]
        t_norm = (t_rel - time_mean) / time_std
        eeg    = item.get("eeg")
 
        if isinstance(eeg, dict):
            eeg_vals = []
            for i, b in enumerate(eeg_bands):
                raw_v   = eeg.get(b, 0.0) or 0.0
                log_v   = math.log1p(float(raw_v))
                normed  = (log_v - eeg_mean[i]) / eeg_std[i]
                eeg_vals.append(normed)
        else:
            eeg_vals = [0.0] * len(eeg_bands)
 
        rows.append([t_norm] + eeg_vals)
 
    x = torch.tensor([rows], dtype=torch.float32)   # [1, seq_len, 9]
    with torch.no_grad():
        return model(x)[0]                        # [384]

def grab_eeg_data():
    response  = requests.get(END_POINT).json()
    return response

def main(arguments = []):
    delay = float(arguments[0]) if len(arguments) > 0 else .1
    model, ckpt = load_model(save_path = "output/model.pt")

    start = time.time()
    while True:
        packet = []
        for i in range(int(1/delay)):
            raw_eeg_data = grab_eeg_data()
            raw_eeg_data['time'] = time.time() - start
            packet.append(raw_eeg_data)
            sleep(delay)
        print(predict(model, packet, ckpt["norm_stats"]))
        

if __name__ == "__main__":
    main(sys.argv[1:])