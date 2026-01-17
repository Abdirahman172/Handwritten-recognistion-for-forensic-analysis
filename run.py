"""
Run:
python run.py --test_dir test --model_keras model.keras --model_h5 model.h5 --labels labels.json --out results.csv
"""

import os
import json
import csv
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from tensorflow.keras.models import load_model

# 🔥 sklearn metrics
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

# =========================
# Arguments
# =========================
p = argparse.ArgumentParser()
p.add_argument("--test_dir", required=True)
p.add_argument("--model_keras", required=True)
p.add_argument("--model_h5", required=True)
p.add_argument("--labels", required=True)
p.add_argument("--out", default="results.csv")
p.add_argument("--img_h", type=int, default=64)
p.add_argument("--img_w", type=int, default=64)
p.add_argument("--min_width", type=int, default=6)
args = p.parse_args()

# =========================
# Load models & labels
# =========================
model1 = load_model(args.model_keras)
model2 = load_model(args.model_h5)

with open(args.labels, "r") as f:
    labels = json.load(f)

idx_to_cls = {i: c for i, c in enumerate(labels)}
cls_to_idx = {c: i for i, c in enumerate(labels)}
n_cls = len(labels)

# =========================
# Utilities (SAME AS TRAIN)
# =========================
def gray(img):
    if img is None:
        return None
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def find_text_lines(g):
    h = g.shape[0]
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    proj = np.sum(closed, axis=1)

    if proj.max() == 0:
        return [(0, h)]

    cut = max(1, int(0.03 * proj.max()))
    lines, inside = [], False
    start = 0

    for y, v in enumerate(proj):
        if v > cut and not inside:
            inside = True
            start = y
        elif v <= cut and inside:
            inside = False
            if y - start >= 6:
                lines.append((start, y))

    if inside:
        lines.append((start, h))

    return lines


def find_words(line):
    g = gray(line)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dil = cv2.dilate(bw, kernel)

    cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w > 8 and h > 8:
            boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[0])
    return [line[y:y+h, x:x+w] for x, y, w, h in boxes]


def split_characters(word, min_w):
    g = gray(word)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    col_sum = np.sum(bw, axis=0)

    if col_sum.max() == 0:
        return []

    th = max(1, int(0.05 * col_sum.max()))
    chars, active = [], False
    start = 0

    for i, v in enumerate(col_sum):
        if v > th and not active:
            active = True
            start = i
        elif v <= th and active:
            active = False
            if i - start >= min_w:
                chars.append(word[:, start:i])

    if active and len(col_sum) - start >= min_w:
        chars.append(word[:, start:])

    return chars


def normalize_char(img, h, w):
    if img is None or img.size == 0:
        return None

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    ih, iw = img.shape[:2]
    if ih == 0 or iw == 0:
        return None

    scale = min(h / ih, w / iw)
    nh = max(1, int(ih * scale))
    nw = max(1, int(iw * scale))

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized

    return canvas.astype(np.float32) / 255.0


# =========================
# Testing
# =========================
y_true, y_pred = [], []
y_score = []   # 🔥 needed for AUC
rows = []

files = [f for f in os.listdir(args.test_dir)
         if f.lower().endswith((".png", ".jpg", ".jpeg"))]

for fname in tqdm(files, desc="Testing"):
    gt = fname[:2]
    img = cv2.imread(os.path.join(args.test_dir, fname))

    if img is None or gt not in cls_to_idx:
        continue

    g = gray(img)
    lines = find_text_lines(g)

    char_imgs = []

    for y1, y2 in lines:
        seg = img[y1:y2, :]
        words = find_words(seg) or [seg]

        for w in words:
            chars = split_characters(w, args.min_width) or [w]
            for c in chars:
                ch = normalize_char(c, args.img_h, args.img_w)
                if ch is not None:
                    char_imgs.append(ch)

    if not char_imgs:
        continue

    X = np.array(char_imgs, dtype=np.float32)

    # 🔥 ENSEMBLE PREDICTION
    p1 = model1.predict(X, verbose=0)
    p2 = model2.predict(X, verbose=0)
    preds = (p1 + p2) / 2.0

    avg_pred = np.sum(preds, axis=0)   # 🔥 per-image probability vector
    pred_idx = np.argmax(avg_pred)
    pred_cls = idx_to_cls[pred_idx]

    rows.append([fname, gt, pred_cls])
    y_true.append(cls_to_idx[gt])
    y_pred.append(pred_idx)
    y_score.append(avg_pred)

# =========================
# Metrics (SKLEARN)
# =========================
y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_score = np.array(y_score)

accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro")

# AUC (One-vs-Rest)
y_true_bin = label_binarize(y_true, classes=list(range(n_cls)))
auc = roc_auc_score(y_true_bin, y_score, average="macro", multi_class="ovr")

# =========================
# Save
# =========================
with open(args.out, "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["filename", "true_class", "predicted_class"])
    wr.writerows(rows)

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Macro F1 : {macro_f1:.4f}")
print(f"Macro AUC: {auc:.4f}")
print("✅ Results saved.")
