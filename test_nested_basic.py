# test_nested_basic.py
# Test/inference for Basic Nested Learning model
# Per-video sequential inference with memory reset, video-level aggregation

import os
import csv
from glob import glob
from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm

# -------------------------
# Config
# -------------------------
DATA_TEST_DIR = "dataset/Test"
NORMAL_DIR = os.path.join(DATA_TEST_DIR, "NormalVideos")
ACC_DIR = os.path.join(DATA_TEST_DIR, "RoadAccidents")

CHECKPOINT_PATH = "best_nested_basic.pth"
SEQ_LEN = 16
IMG_SIZE = 64
BATCH_SIZE = 8
NUM_WORKERS = 0
OUT_CSV = "test_nested_predictions.csv"
PROB_THRESHOLD = 0.25  # for mean_prob video-level aggregation

# Device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print("Device:", DEVICE)

# -------------------------
# Helpers
# -------------------------
def parse_video_id_and_index(fname: str):
    base = os.path.basename(fname)
    if "_" not in base:
        return os.path.splitext(base)[0], 0
    prefix, suffix = base.rsplit("_", 1)
    idx_str = os.path.splitext(suffix)[0]
    try:
        idx = int(idx_str)
    except:
        idx = 0
    return prefix, idx

def build_video_index(folder_path: str):
    files = glob(os.path.join(folder_path, "*.png"))
    vids = defaultdict(list)
    for f in files:
        vid, idx = parse_video_id_and_index(f)
        vids[vid].append((idx, f))
    for k in list(vids.keys()):
        vids[k].sort(key=lambda x: x[0])
        vids[k] = [fp for _, fp in vids[k]]
    return vids

def sequences_from_video(frame_paths: List[str], seq_len: int) -> List[List[str]]:
    n = len(frame_paths)
    if n == 0:
        return []
    sequences = []
    if n >= seq_len:
        stride = seq_len
        for start in range(0, n - seq_len + 1, stride):
            sequences.append(frame_paths[start:start + seq_len])
        if (n - seq_len) % stride != 0:
            sequences.append(frame_paths[-seq_len:])
    else:
        pad_needed = seq_len - n
        seq = frame_paths + [frame_paths[-1]] * pad_needed
        sequences.append(seq)
    return sequences

# -------------------------
# Dataset
# -------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
])

class VideoSequenceDataset(Dataset):
    def __init__(self, seq_list):
        self.seq_list = seq_list
    def __len__(self):
        return len(self.seq_list)
    def __getitem__(self, idx):
        frame_paths, label, vid = self.seq_list[idx]
        frames = []
        for p in frame_paths:
            img = Image.open(p).convert("RGB")
            img = transform(img)
            frames.append(img)
        frames_t = torch.stack(frames, dim=0)
        return frames_t, torch.tensor(label, dtype=torch.long), vid

# -------------------------
# Model (must match training architecture exactly)
# -------------------------
class SmallCNNEncoder(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, emb_dim),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)


class AdaptiveMemory(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha_proj = nn.Linear(hidden_dim, 1)
        self.medium_decay = nn.Parameter(torch.tensor(0.9))
        self.memories = {}

    def forward(self, h_last, vids: List[str]):
        device = h_last.device
        B = h_last.size(0)
        alpha = torch.sigmoid(self.alpha_proj(h_last))
        mem_batch = []
        for i in range(B):
            vid = vids[i]
            h = h_last[i:i+1]
            if vid not in self.memories:
                self.memories[vid] = torch.zeros_like(h)
            M = self.memories[vid]
            a = alpha[i:i+1]
            M_new = a * M + (1.0 - a) * h
            M_smoothed = self.medium_decay * M + (1.0 - self.medium_decay) * M_new
            self.memories[vid] = M_smoothed.detach()
            mem_batch.append(self.memories[vid])
        return torch.cat(mem_batch, dim=0)

    def reset(self, vids=None):
        if vids is None:
            self.memories = {}
        else:
            for v in vids:
                self.memories.pop(v, None)


class CNN_LSTM_NestedBasic(nn.Module):
    def __init__(self, emb_dim=256, hidden_dim=256, num_layers=1, num_classes=2):
        super().__init__()
        self.encoder = SmallCNNEncoder(emb_dim=emb_dim)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.adaptive_memory = AdaptiveMemory(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, vids):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        emb = self.encoder(x)
        emb = emb.view(B, T, -1)
        out, _ = self.lstm(emb)
        h_last = out[:, -1, :]
        mem = self.adaptive_memory(h_last, vids)
        fused = torch.cat([h_last, mem], dim=1)
        return self.classifier(fused)

# -------------------------
# Build test index
# -------------------------
print("Indexing test dataset...")
normal_vids = build_video_index(NORMAL_DIR)
acc_vids = build_video_index(ACC_DIR)
print(f"Found {len(normal_vids)} normal videos, {len(acc_vids)} accident videos (test).")

# -------------------------
# Load model
# -------------------------
assert os.path.exists(CHECKPOINT_PATH), f"Checkpoint not found: {CHECKPOINT_PATH}"
model = CNN_LSTM_NestedBasic(emb_dim=256, hidden_dim=256, num_layers=1, num_classes=2).to(DEVICE)
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"Loaded checkpoint (epoch {ckpt.get('epoch','?')}, phase {ckpt.get('phase','?')}, val_f1={ckpt.get('val_f1','?')})")

# -------------------------
# Per-video sequential inference (memory accumulates across sequences of same video)
# -------------------------
softmax = nn.Softmax(dim=1)
all_video_results = []  # list of (vid, true_label, mean_prob, pred)

all_videos = []
for vid, frames in normal_vids.items():
    all_videos.append((vid, frames, 0))
for vid, frames in acc_vids.items():
    all_videos.append((vid, frames, 1))

print("Running per-video inference...")
with torch.no_grad():
    for vid, frames, true_label in tqdm(all_videos, desc="Videos"):
        model.adaptive_memory.reset()
        seqs = sequences_from_video(frames, SEQ_LEN)
        seq_probs = []

        for seq_frames in seqs:
            imgs = []
            for f in seq_frames:
                img = Image.open(f).convert("RGB")
                img = transform(img)
                imgs.append(img)
            imgs_t = torch.stack(imgs, dim=0).unsqueeze(0).to(DEVICE)  # [1, T, C, H, W]
            logits = model(imgs_t, [vid])
            prob = softmax(logits)[0, 1].item()
            seq_probs.append(prob)

        mean_prob = sum(seq_probs) / len(seq_probs) if seq_probs else 0.0
        pred = 1 if mean_prob >= PROB_THRESHOLD else 0
        all_video_results.append((vid, true_label, mean_prob, pred))

# -------------------------
# Metrics
# -------------------------
v_true = [r[1] for r in all_video_results]
v_pred = [r[3] for r in all_video_results]
v_probs = [r[2] for r in all_video_results]

v_acc = accuracy_score(v_true, v_pred)
v_prec, v_rec, v_f1, _ = precision_recall_fscore_support(v_true, v_pred, average='binary', zero_division=0)
v_cm = confusion_matrix(v_true, v_pred)

print(f"\n=== Video-level Results (threshold={PROB_THRESHOLD}) ===")
print(f"Accuracy:  {v_acc:.4f}")
print(f"Precision: {v_prec:.4f}")
print(f"Recall:    {v_rec:.4f}")
print(f"F1:        {v_f1:.4f}")
print(f"Confusion Matrix:\n{v_cm}")

# Also eval at other thresholds for reference
for thresh in [0.25, 0.4, 0.5]:
    preds_t = [1 if p >= thresh else 0 for p in v_probs]
    acc_t = accuracy_score(v_true, preds_t)
    _, _, f1_t, _ = precision_recall_fscore_support(v_true, preds_t, average='binary', zero_division=0)
    print(f"  threshold={thresh:.2f} => acc={acc_t:.4f} f1={f1_t:.4f}")

# -------------------------
# Save predictions CSV
# -------------------------
with open(OUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['video_id', 'true_label', 'mean_prob', 'predicted'])
    for vid, true_label, mean_prob, pred in sorted(all_video_results, key=lambda x: x[0]):
        writer.writerow([vid, true_label, f"{mean_prob:.4f}", pred])
print(f"Saved predictions to {OUT_CSV}")
