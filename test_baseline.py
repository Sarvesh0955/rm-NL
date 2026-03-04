# test_baseline.py
# Run inference on the Test split and produce sequence-level and video-level metrics.
import os
import random
from glob import glob
from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import csv
from tqdm import tqdm

# -----------------------
# Config (edit if needed)
# -----------------------
DATA_TEST_DIR = "dataset/Test"   # directory that contains Test/NormalVideos and Test/RoadAccidents
NORMAL_DIR = os.path.join(DATA_TEST_DIR, "NormalVideos")
ACC_DIR = os.path.join(DATA_TEST_DIR, "RoadAccidents")
CHECKPOINT_PATH = "best_baseline_checkpoint.pth"
BATCH_SIZE = 8
SEQ_LEN = 16
IMG_SIZE = 64
NUM_WORKERS = 0  # use 0 on macOS to avoid multiprocessing issues
AGGREGATION = "mean_prob"  # "majority" or "mean_prob" (video-level aggregation)
OUT_CSV = "test_predictions.csv"

# device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print("Device:", DEVICE)

# -----------------------
# Helpers: parse filenames
# -----------------------
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
    vids = {}
    for f in files:
        vid, idx = parse_video_id_and_index(f)
        vids.setdefault(vid, []).append((idx, f))
    # sort frames per video
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
        stride = seq_len  # non-overlap
        for start in range(0, n - seq_len + 1, stride):
            sequences.append(frame_paths[start:start + seq_len])
        if (n - seq_len) % stride != 0:
            sequences.append(frame_paths[-seq_len:])
    else:
        pad_needed = seq_len - n
        seq = frame_paths + [frame_paths[-1]] * pad_needed
        sequences.append(seq)
    return sequences

# -----------------------
# Dataset
# -----------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45,0.45,0.45], std=[0.225,0.225,0.225]),
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
        frames_t = torch.stack(frames, dim=0)  # [T,C,H,W]
        return frames_t, torch.tensor(label, dtype=torch.long), vid

# -----------------------
# Model definition (same as training)
# -----------------------
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
            nn.AdaptiveAvgPool2d((4,4)),
            nn.Flatten(),
            nn.Linear(128*4*4, emb_dim),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)

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
        x = x.view(B*T, C, H, W)
        emb = self.encoder(x)
        emb = emb.view(B, T, -1)
        out, (hn, cn) = self.lstm(emb)
        last = out[:, -1, :]
        logits = self.classifier(last)
        return logits

# -----------------------
# Build test index
# -----------------------
print("Indexing test dataset...")
normal_vids = build_video_index(NORMAL_DIR)
acc_vids = build_video_index(ACC_DIR)
print(f"Found {len(normal_vids)} normal videos, {len(acc_vids)} accident videos (test).")

seq_index = []
for vid, frames in acc_vids.items():
    seqs = sequences_from_video(frames, SEQ_LEN)
    for s in seqs:
        seq_index.append((s, 1, vid))
for vid, frames in normal_vids.items():
    seqs = sequences_from_video(frames, SEQ_LEN)
    for s in seqs:
        seq_index.append((s, 0, vid))

print("Total test sequences:", len(seq_index))
test_dataset = VideoSequenceDataset(seq_index)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# -----------------------
# Load model checkpoint
# -----------------------
assert os.path.exists(CHECKPOINT_PATH), f"Checkpoint not found: {CHECKPOINT_PATH}"
model = CNN_LSTM_Classifier(emb_dim=256, hidden_dim=256, num_layers=1, num_classes=2).to(DEVICE)
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()
print("Loaded checkpoint, starting inference...")

# -----------------------
# Run inference (sequence-level)
# -----------------------
all_preds = []
all_labels = []
all_probs = []
all_vids = []

softmax = nn.Softmax(dim=1)
with torch.no_grad():
    for seqs, labels, vids in tqdm(test_loader, desc="Testing"):
        seqs = seqs.to(DEVICE)  # [B,T,C,H,W]
        logits = model(seqs)    # [B,2]
        probs = softmax(logits)[:,1].cpu().numpy().tolist()  # probability of accident (class 1)
        preds = logits.argmax(dim=1).cpu().numpy().tolist()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy().tolist())
        all_vids.extend(vids)

# Sequence-level metrics
acc = accuracy_score(all_labels, all_preds)
prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
cm = confusion_matrix(all_labels, all_preds)
print("Sequence-level metrics:")
print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
print("Confusion matrix:\n", cm)

# -----------------------
# Video-level aggregation
# -----------------------
vid_scores = defaultdict(list)   # vid -> list of probs
vid_labels = {}                  # vid -> true label (from any sequence)
for v, lab, p in zip(all_vids, all_labels, all_probs):
    vid_scores[v].append(p)
    vid_labels[v] = lab

vid_preds = {}
vid_prob = {}
for v, probs in vid_scores.items():
    if AGGREGATION == "majority":
        # convert per-seq probs to binary with threshold 0.5 then majority vote
        seq_bin = [1 if p >= 0.5 else 0 for p in probs]
        pred = 1 if sum(seq_bin) > len(seq_bin)/2 else 0
        vid_preds[v] = pred
        vid_prob[v] = sum(probs) / len(probs)
    else:  # mean_prob
        mean_p = sum(probs) / len(probs)
        pred = 1 if mean_p >= 0.25 else 0
        vid_preds[v] = pred
        vid_prob[v] = mean_p

# Collect video-level arrays
v_true = []
v_pred = []
v_prob_list = []
v_ids = []
for v, true_lab in vid_labels.items():
    v_true.append(true_lab)
    v_pred.append(vid_preds[v])
    v_prob_list.append(vid_prob[v])
    v_ids.append(v)

# video-level metrics
v_acc = accuracy_score(v_true, v_pred)
v_prec, v_rec, v_f1, _ = precision_recall_fscore_support(v_true, v_pred, average='binary', zero_division=0)
v_cm = confusion_matrix(v_true, v_pred)
print("\nVideo-level metrics (aggregation=%s):" % AGGREGATION)
print(f"Accuracy: {v_acc:.4f}, Precision: {v_prec:.4f}, Recall: {v_rec:.4f}, F1: {v_f1:.4f}")
print("Confusion matrix:\n", v_cm)

# -----------------------
# Save per-video predictions to CSV
# -----------------------
with open(OUT_CSV, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["video_id", "true_label", "pred_label", "prob_accident"])
    for vid, true_lab in vid_labels.items():
        writer.writerow([vid, true_lab, vid_preds[vid], vid_prob[vid]])

print(f"Saved per-video predictions to {OUT_CSV}")