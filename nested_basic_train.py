# nested_basic_train.py
# Basic Nested Learning: CNN+LSTM (from baseline) + Adaptive Memory module
# Two-phase training: Phase 1 trains memory+classifier only, Phase 2 fine-tunes all
# Loads pre-trained baseline weights from best_baseline_checkpoint.pth

import os
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
DATA_DIR = "dataset/Train"
NORMAL_DIR = os.path.join(DATA_DIR, "NormalVideos")
ACC_DIR = os.path.join(DATA_DIR, "RoadAccidents")

MAX_NORMAL_VIDEOS = 200
SEQ_LEN = 16
IMG_SIZE = 64
BATCH_SIZE = 8
RANDOM_SEED = 42
UNDERSAMPLE_RATIO = 3
VAL_SPLIT = 0.15
NUM_WORKERS = 0

BASELINE_CKPT = "best_baseline_checkpoint.pth"
SAVE_PATH = "best_nested_basic.pth"

# Phase 1: train memory + classifier only (backbone frozen)
PHASE1_EPOCHS = 5
PHASE1_LR = 1e-3
# Phase 2: fine-tune everything
PHASE2_EPOCHS = 10
PHASE2_LR = 1e-4
# Loss class weights
LOSS_WEIGHT = [1.0, 25.0]

# Device
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

# -------------------------
# Sequence creation
# -------------------------
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
# Index dataset
# -------------------------
print("Indexing dataset...")
normal_vids = build_video_index(NORMAL_DIR)
acc_vids = build_video_index(ACC_DIR)
print(f"Found {len(normal_vids)} normal videos, {len(acc_vids)} accident videos.")

normal_keys = list(normal_vids.keys())
random.seed(RANDOM_SEED)
random.shuffle(normal_keys)
if len(normal_keys) > MAX_NORMAL_VIDEOS:
    normal_keys = normal_keys[:MAX_NORMAL_VIDEOS]
normal_vids = {k: normal_vids[k] for k in normal_keys}
print(f"Using {len(normal_vids)} normal videos, {len(acc_vids)} accident videos.")

# -------------------------
# Train/Val split (video-level)
# -------------------------
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

def build_sequences_from_video_sets(normal_vids, acc_vids, train_norm, train_acc, val_norm, val_acc, seq_len):
    train_list, val_list = [], []
    for vid, frames in acc_vids.items():
        seqs = sequences_from_video(frames, seq_len)
        for s in seqs:
            (train_list if vid in train_acc else val_list).append((s, 1, vid))
    for vid, frames in normal_vids.items():
        seqs = sequences_from_video(frames, seq_len)
        for s in seqs:
            (train_list if vid in train_norm else val_list).append((s, 0, vid))
    return train_list, val_list

train_sequences, val_sequences = build_sequences_from_video_sets(
    normal_vids, acc_vids, train_norm, train_acc, val_norm, val_acc, SEQ_LEN
)
print(f"Train sequences: {len(train_sequences)}, Val sequences: {len(val_sequences)}")
print(f"Train: Accident {sum(1 for s in train_sequences if s[1]==1)}, Normal {sum(1 for s in train_sequences if s[1]==0)}")

# -------------------------
# Dataset and transforms
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

train_dataset = VideoSequenceDataset(train_sequences)
val_dataset = VideoSequenceDataset(val_sequences)

# -------------------------
# Per-epoch undersampled loader
# -------------------------
def make_epoch_sampler(train_seq_list, ratio=UNDERSAMPLE_RATIO, seed=None):
    if seed is not None:
        random.seed(seed)
    acc_indices = [i for i, item in enumerate(train_seq_list) if item[1] == 1]
    norm_indices = [i for i, item in enumerate(train_seq_list) if item[1] == 0]
    n_acc = len(acc_indices)
    desired_norm = min(len(norm_indices), ratio * max(1, n_acc))
    sampled_norm = random.sample(norm_indices, desired_norm) if desired_norm < len(norm_indices) else norm_indices.copy()
    chosen = acc_indices + sampled_norm
    random.shuffle(chosen)
    return chosen

def make_train_loader(epoch_idx):
    indices = make_epoch_sampler(train_sequences, ratio=UNDERSAMPLE_RATIO, seed=RANDOM_SEED + epoch_idx)
    sampler = SubsetRandomSampler(indices)
    return DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=False)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

# -------------------------
# Model: SmallCNN + LSTM + AdaptiveMemory (Basic Nested Learning)
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
    """Per-video adaptive memory with learned gate and medium-timescale smoothing."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha_proj = nn.Linear(hidden_dim, 1)
        self.medium_decay = nn.Parameter(torch.tensor(0.9))
        self.memories = {}

    def forward(self, h_last, vids: List[str]):
        """
        h_last: [B, hidden_dim]
        vids: list of video ids (length B)
        returns: [B, hidden_dim] memory tensor
        """
        device = h_last.device
        B = h_last.size(0)
        alpha = torch.sigmoid(self.alpha_proj(h_last))  # [B, 1]
        mem_batch = []

        for i in range(B):
            vid = vids[i]
            h = h_last[i:i+1]  # [1, hidden_dim]
            if vid not in self.memories:
                self.memories[vid] = torch.zeros_like(h)
            M = self.memories[vid]
            a = alpha[i:i+1]  # [1, 1]
            M_new = a * M + (1.0 - a) * h
            M_smoothed = self.medium_decay * M + (1.0 - self.medium_decay) * M_new
            self.memories[vid] = M_smoothed.detach()
            mem_batch.append(self.memories[vid])

        return torch.cat(mem_batch, dim=0)  # [B, hidden_dim]

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
        emb = self.encoder(x)          # [B*T, emb_dim]
        emb = emb.view(B, T, -1)       # [B, T, emb_dim]
        out, _ = self.lstm(emb)         # [B, T, hidden]
        h_last = out[:, -1, :]          # [B, hidden]
        mem = self.adaptive_memory(h_last, vids)  # [B, hidden]
        fused = torch.cat([h_last, mem], dim=1)   # [B, 2*hidden]
        return self.classifier(fused)

# -------------------------
# Create model and load baseline weights
# -------------------------
model = CNN_LSTM_NestedBasic(emb_dim=256, hidden_dim=256, num_layers=1, num_classes=2).to(DEVICE)
print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Load baseline encoder + lstm weights
assert os.path.exists(BASELINE_CKPT), f"Baseline checkpoint not found: {BASELINE_CKPT}"
baseline_ckpt = torch.load(BASELINE_CKPT, map_location=DEVICE)
baseline_state = baseline_ckpt['model_state']

# Map matching keys (encoder.* and lstm.*)
model_state = model.state_dict()
loaded_keys = []
skipped_keys = []
for k, v in baseline_state.items():
    if k in model_state and model_state[k].shape == v.shape:
        model_state[k] = v
        loaded_keys.append(k)
    else:
        skipped_keys.append(k)

model.load_state_dict(model_state)
print(f"Loaded {len(loaded_keys)} keys from baseline: encoder={sum(1 for k in loaded_keys if k.startswith('encoder'))}, lstm={sum(1 for k in loaded_keys if k.startswith('lstm'))}")
print(f"Skipped {len(skipped_keys)} keys (shape mismatch or new): {skipped_keys}")

# -------------------------
# Metrics
# -------------------------
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return acc, p, r, f1, cm

# -------------------------
# Training function (one epoch)
# -------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_pred, all_true = [], []
    pbar = tqdm(loader, desc="  train")
    for seqs, labels, vids in pbar:
        vids = [str(v) for v in vids]
        seqs = seqs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(seqs, vids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * seqs.size(0)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_pred.extend(preds.tolist())
        all_true.extend(labels.detach().cpu().numpy().tolist())
        pbar.set_postfix(loss=total_loss / (len(all_true) + 1e-12))
    return total_loss, all_true, all_pred

def validate(model, loader, criterion, device):
    model.eval()
    model.adaptive_memory.reset()
    val_loss = 0.0
    all_pred, all_true = [], []
    with torch.no_grad():
        for seqs, labels, vids in tqdm(loader, desc="  val"):
            vids = [str(v) for v in vids]
            seqs = seqs.to(device)
            labels = labels.to(device)
            logits = model(seqs, vids)
            loss = criterion(logits, labels)
            val_loss += loss.item() * seqs.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_pred.extend(preds.tolist())
            all_true.extend(labels.detach().cpu().numpy().tolist())
    return val_loss, all_true, all_pred

# -------------------------
# Two-phase training
# -------------------------
criterion = nn.CrossEntropyLoss(weight=torch.tensor(LOSS_WEIGHT, device=DEVICE))
best_val_f1 = 0.0

# === Phase 1: Freeze encoder + lstm, train memory + classifier ===
print("\n=== Phase 1: Training memory + classifier (backbone frozen) ===")
for param in model.encoder.parameters():
    param.requires_grad = False
for param in model.lstm.parameters():
    param.requires_grad = False
trainable_p1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Phase 1 trainable params: {trainable_p1}")

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=PHASE1_LR)

for epoch in range(1, PHASE1_EPOCHS + 1):
    model.adaptive_memory.reset()
    train_loader = make_train_loader(epoch)
    total_loss, y_true, y_pred = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    t_acc, t_p, t_r, t_f1, _ = compute_metrics(y_true, y_pred)
    print(f"Phase1 Epoch {epoch}/{PHASE1_EPOCHS} TRAIN loss={total_loss/len(train_loader.dataset):.4f} acc={t_acc:.4f} f1={t_f1:.4f}")

    val_loss, v_true, v_pred = validate(model, val_loader, criterion, DEVICE)
    v_acc, v_p, v_r, v_f1, v_cm = compute_metrics(v_true, v_pred)
    print(f"  VAL loss={val_loss/len(val_loader.dataset):.4f} acc={v_acc:.4f} prec={v_p:.4f} rec={v_r:.4f} f1={v_f1:.4f}")
    print(f"  CM:\n{v_cm}")

    if v_f1 > best_val_f1:
        best_val_f1 = v_f1
        torch.save({'epoch': epoch, 'phase': 1, 'model_state': model.state_dict(), 'val_f1': v_f1}, SAVE_PATH)
        print("  Saved new best model.")

# === Phase 2: Unfreeze all, fine-tune with lower LR ===
print("\n=== Phase 2: Fine-tuning all parameters ===")
for param in model.encoder.parameters():
    param.requires_grad = True
for param in model.lstm.parameters():
    param.requires_grad = True
trainable_p2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Phase 2 trainable params: {trainable_p2}")

optimizer = torch.optim.Adam(model.parameters(), lr=PHASE2_LR)

for epoch in range(1, PHASE2_EPOCHS + 1):
    model.adaptive_memory.reset()
    train_loader = make_train_loader(PHASE1_EPOCHS + epoch)
    total_loss, y_true, y_pred = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    t_acc, t_p, t_r, t_f1, _ = compute_metrics(y_true, y_pred)
    print(f"Phase2 Epoch {epoch}/{PHASE2_EPOCHS} TRAIN loss={total_loss/len(train_loader.dataset):.4f} acc={t_acc:.4f} f1={t_f1:.4f}")

    val_loss, v_true, v_pred = validate(model, val_loader, criterion, DEVICE)
    v_acc, v_p, v_r, v_f1, v_cm = compute_metrics(v_true, v_pred)
    print(f"  VAL loss={val_loss/len(val_loader.dataset):.4f} acc={v_acc:.4f} prec={v_p:.4f} rec={v_r:.4f} f1={v_f1:.4f}")
    print(f"  CM:\n{v_cm}")

    if v_f1 > best_val_f1:
        best_val_f1 = v_f1
        torch.save({'epoch': PHASE1_EPOCHS + epoch, 'phase': 2, 'model_state': model.state_dict(), 'val_f1': v_f1}, SAVE_PATH)
        print("  Saved new best model.")

print(f"\nTraining complete. Best val F1: {best_val_f1:.4f}")
