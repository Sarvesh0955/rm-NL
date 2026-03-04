# baseline_video_accident.py
# Baseline: small CNN encoder + LSTM temporal classifier
# - Expects dataset/NormalVideos and dataset/RoadAccidents
# - Filenames like: video001_0.png, video001_1.png, ...
# - Uses per-video grouping and forms sequences of seq_len frames
# - Undersamples Normal per epoch to reduce imbalance

import os
import re
import random
from glob import glob
from collections import defaultdict
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

# -------------------------
# Config / Hyperparameters
# -------------------------
DATA_DIR = "dataset/Train"   # change if needed
NORMAL_DIR = os.path.join(DATA_DIR, "NormalVideos")
ACC_DIR = os.path.join(DATA_DIR, "RoadAccidents")

SEQ_LEN = 16           # frames per sequence
IMG_SIZE = 64          # frames are already 64x64
BATCH_SIZE = 8         # safe for M4; increase if GPU/MPS is fast and memory allows
LR = 1e-3
EPOCHS = 20
RANDOM_SEED = 42
UNDERSAMPLE_RATIO = 3   # normal:accident ratio per epoch (set to 3)
VAL_SPLIT = 0.15        # fraction of videos for validation
MAX_NORMAL_VIDEOS = 200

DEVICE = None
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Device:", DEVICE)

# -------------------------
# Helpers: parse filenames
# -------------------------
# We expect filenames with a "video id" and frame number separated by underscore,
# e.g., "video001_0.png" or "somevideo_000.png". We'll extract the prefix before last underscore.
def parse_video_id_and_index(fname: str):
    # fname may be '.../video001_12.png'
    base = os.path.basename(fname)
    # find last underscore
    if "_" not in base:
        # fallback: treat the whole basename as video id
        return os.path.splitext(base)[0], 0
    prefix, suffix = base.rsplit("_", 1)
    # suffix might be '12.png' -> strip ext
    idx_str = os.path.splitext(suffix)[0]
    try:
        idx = int(idx_str)
    except:
        idx = 0
    return prefix, idx

# -------------------------
# Build per-video indices
# -------------------------
def build_video_index(folder_path: str):
    files = glob(os.path.join(folder_path, "*.png"))
    vids = defaultdict(list)
    for f in files:
        vid, idx = parse_video_id_and_index(f)
        vids[vid].append((idx, f))
    # sort frames per video by index
    for k in list(vids.keys()):
        vids[k].sort(key=lambda x: x[0])
        vids[k] = [fp for _, fp in vids[k]]
    return vids  # dict: vid -> list of frame paths

print("Indexing dataset... (this may take a moment)")
normal_vids = build_video_index(NORMAL_DIR)
acc_vids = build_video_index(ACC_DIR)

normal_keys = list(normal_vids.keys())
random.shuffle(normal_keys)

if len(normal_keys) > MAX_NORMAL_VIDEOS:
    normal_keys = normal_keys[:MAX_NORMAL_VIDEOS]

normal_vids = {k: normal_vids[k] for k in normal_keys}

print("Normal videos used:", len(normal_vids))
print("Accident videos used:", len(acc_vids))
print(f"Found {len(normal_vids)} normal videos, {len(acc_vids)} accident videos.")

# -------------------------
# Build video -> sequences mapping
# -------------------------
def sequences_from_video(frame_paths: List[str], seq_len: int) -> List[List[str]]:
    """Return list of sequences (lists of file paths) for given video frames.
       We'll sample non-overlapping sliding windows of length seq_len when possible,
       otherwise overlapping windows. For very short videos, we will pad by repeating last frame."""
    n = len(frame_paths)
    if n == 0:
        return []
    sequences = []
    if n >= seq_len:
        # sample non-overlapping windows (stride = seq_len) first
        stride = seq_len
        for start in range(0, n - seq_len + 1, stride):
            sequences.append(frame_paths[start:start + seq_len])
        # if leftover frames at end, also include final window (overlap)
        if (n - seq_len) % stride != 0:
            sequences.append(frame_paths[-seq_len:])
    else:
        # pad by repeating last frame
        pad_needed = seq_len - n
        seq = frame_paths + [frame_paths[-1]] * pad_needed
        sequences.append(seq)
    return sequences

# Create dataset index: a flat list of (sequence_paths, label)
def build_sequence_index(normal_vids, acc_vids, seq_len):
    seq_index = []
    # accidents label=1, normal label=0
    for vid, frames in acc_vids.items():
        seqs = sequences_from_video(frames, seq_len)
        for s in seqs:
            seq_index.append((s, 1, vid))
    for vid, frames in normal_vids.items():
        seqs = sequences_from_video(frames, seq_len)
        for s in seqs:
            seq_index.append((s, 0, vid))
    return seq_index

print("Building sequence index (may take a while depending on number of frames)...")
sequence_index = build_sequence_index(normal_vids, acc_vids, SEQ_LEN)
print(f"Total sequences: {len(sequence_index)}")
# count per class
num_acc = sum(1 for s in sequence_index if s[1] == 1)
num_norm = sum(1 for s in sequence_index if s[1] == 0)
print(f"Accident sequences: {num_acc}, Normal sequences: {num_norm}")

# -------------------------
# Train/Val split at video-level
# -------------------------
# We must split videos (not sequences) to avoid leakage.
def split_videos(normal_vids, acc_vids, val_split=0.15, seed=RANDOM_SEED):
    random.seed(seed)
    normal_list = list(normal_vids.keys())
    acc_list = list(acc_vids.keys())
    random.shuffle(normal_list)
    random.shuffle(acc_list)
    n_val_norm = max(1, int(len(normal_list) * val_split))
    n_val_acc = max(1, int(len(acc_list) * val_split))
    val_norm = set(normal_list[:n_val_norm])
    val_acc = set(acc_list[:n_val_acc])
    train_norm = set(normal_list[n_val_norm:])
    train_acc = set(acc_list[n_val_acc:])
    return train_norm, train_acc, val_norm, val_acc

train_norm, train_acc, val_norm, val_acc = split_videos(normal_vids, acc_vids, VAL_SPLIT)
print(f"Train videos: normal {len(train_norm)}, acc {len(train_acc)} | Val videos: normal {len(val_norm)}, acc {len(val_acc)}")

# Build sequence lists for train/val
def build_sequences_from_video_sets(normal_vids, acc_vids, train_norm, train_acc, val_norm, val_acc, seq_len):
    train_list = []
    val_list = []
    for vid, frames in acc_vids.items():
        seqs = sequences_from_video(frames, seq_len)
        for s in seqs:
            if vid in train_acc:
                train_list.append((s, 1, vid))
            else:
                val_list.append((s, 1, vid))
    for vid, frames in normal_vids.items():
        seqs = sequences_from_video(frames, seq_len)
        for s in seqs:
            if vid in train_norm:
                train_list.append((s, 0, vid))
            else:
                val_list.append((s, 0, vid))
    return train_list, val_list

train_sequences, val_sequences = build_sequences_from_video_sets(normal_vids, acc_vids, train_norm, train_acc, val_norm, val_acc, SEQ_LEN)
print(f"Train sequences: {len(train_sequences)}, Val sequences: {len(val_sequences)}")
train_acc_count = sum(1 for s in train_sequences if s[1] == 1)
train_norm_count = sum(1 for s in train_sequences if s[1] == 0)
print(f"Train: Accident {train_acc_count}, Normal {train_norm_count}")

# -------------------------
# Dataset Class
# -------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),                 # converts to [0,1]
    # images are small; we do simple normalization (mean/std approximate)
    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
])

class VideoSequenceDataset(Dataset):
    def __init__(self, seq_list):
        """
        seq_list: list of tuples (list_of_frame_paths, label, video_id)
        """
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
        # frames: list of [C,H,W] tensors => stack to [T,C,H,W]
        frames_t = torch.stack(frames, dim=0)
        return frames_t, torch.tensor(label, dtype=torch.long), vid

# -------------------------
# Small CNN Encoder + LSTM Classifier
# -------------------------
class SmallCNNEncoder(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()
        # keep it small (64x64 input)
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 64x64
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4)),  # 4x4
            nn.Flatten(),
            nn.Linear(128*4*4, emb_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: [B, C, H, W]
        return self.net(x)  # [B, emb_dim]

class CNN_LSTM_Classifier(nn.Module):
    def __init__(self, emb_dim=256, hidden_dim=256, num_layers=1, num_classes=2):
        super().__init__()
        self.encoder = SmallCNNEncoder(emb_dim=emb_dim)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, num_classes)
        )

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        emb = self.encoder(x)  # [B*T, emb_dim]
        emb = emb.view(B, T, -1)  # [B, T, emb_dim]
        out, (hn, cn) = self.lstm(emb)  # out: [B,T,hidden]
        last = out[:, -1, :]  # [B, hidden]
        logits = self.classifier(last)
        return logits

# -------------------------
# Create datasets
# -------------------------
train_dataset = VideoSequenceDataset(train_sequences)
val_dataset = VideoSequenceDataset(val_sequences)

# Helper to build per-epoch undersampled index for training
def make_epoch_sampler(train_seq_list, ratio=UNDERSAMPLE_RATIO, seed=None):
    """
    train_seq_list: list of tuples (seq_paths, label, vid)
    We'll undersample Normal sequences so that Normal:Accident ~ ratio.
    Returns a list of indices to use for this epoch (randomized).
    """
    if seed is not None:
        random.seed(seed)
    acc_indices = [i for i, item in enumerate(train_seq_list) if item[1] == 1]
    norm_indices = [i for i, item in enumerate(train_seq_list) if item[1] == 0]
    n_acc = len(acc_indices)
    # desired normal count:
    desired_norm = min(len(norm_indices), ratio * max(1, n_acc))
    sampled_norm = random.sample(norm_indices, desired_norm) if desired_norm < len(norm_indices) else norm_indices.copy()
    chosen = acc_indices + sampled_norm
    random.shuffle(chosen)
    return chosen

# small function to build DataLoader for a given epoch (to vary sampler each epoch)
def make_train_loader(epoch_idx):
    indices = make_epoch_sampler(train_sequences, ratio=UNDERSAMPLE_RATIO, seed=RANDOM_SEED + epoch_idx)
    sampler = SubsetRandomSampler(indices)
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=False)
    return loader

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

# -------------------------
# Training utilities
# -------------------------
def compute_metrics_all(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return acc, p, r, f1, cm

# -------------------------
# Training loop
# -------------------------
model = CNN_LSTM_Classifier(emb_dim=256, hidden_dim=256, num_layers=1, num_classes=2).to(DEVICE)
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor([1.0, 30.0]).to(DEVICE)
)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print("Model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

best_val_f1 = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loader = make_train_loader(epoch)
    total_loss = 0.0
    all_pred = []
    all_true = []
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} train")
    for batch in pbar:
        seqs, labels, vids = batch
        # seqs: [B, T, C, H, W]
        seqs = seqs.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(seqs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * seqs.size(0)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_pred.extend(preds.tolist())
        all_true.extend(labels.detach().cpu().numpy().tolist())
        pbar.set_postfix(loss=total_loss / (len(all_true) + 1e-12))

    train_acc, train_p, train_r, train_f1, train_cm = compute_metrics_all(all_true, all_pred)
    print(f"Epoch {epoch} TRAIN loss={total_loss/len(train_loader.dataset):.4f} acc={train_acc:.4f} f1={train_f1:.4f}")

    # Validation
    model.eval()
    val_preds = []
    val_trues = []
    val_loss = 0.0
    with torch.no_grad():
        for seqs, labels, vids in tqdm(val_loader, desc="Val"):
            seqs = seqs.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(seqs)
            loss = criterion(logits, labels)
            val_loss += loss.item() * seqs.size(0)
            pred = logits.argmax(dim=1).cpu().numpy()
            val_preds.extend(pred.tolist())
            val_trues.extend(labels.detach().cpu().numpy().tolist())

    val_acc, val_p, val_r, val_f1, val_cm = compute_metrics_all(val_trues, val_preds)
    avg_val_loss = val_loss / len(val_loader.dataset)
    print(f"VAL  loss={avg_val_loss:.4f} acc={val_acc:.4f} precision={val_p:.4f} recall={val_r:.4f} f1={val_f1:.4f}")
    print("Confusion matrix:\n", val_cm)

    # Save best
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_f1': val_f1
        }, "best_baseline_checkpoint.pth")
        print("Saved new best model.")

print("Training complete. Best val F1:", best_val_f1)