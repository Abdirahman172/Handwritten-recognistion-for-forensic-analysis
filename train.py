import os
import json
import random
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import argparse

# =========================
# Arguments
# =========================
argp = argparse.ArgumentParser()
argp.add_argument("--train_dir", default="train")
argp.add_argument("--model_out", default="model.keras")
argp.add_argument("--labels_out", default="labels.json")
argp.add_argument("--img_h", type=int, default=64)
argp.add_argument("--img_w", type=int, default=64)
argp.add_argument("--min_width", type=int, default=6)
argp.add_argument("--epochs", type=int, default=85)
argp.add_argument("--batch", type=int, default=64)
argp.add_argument("--val_ratio", type=float, default=0.10)
argp.add_argument("--seed", type=int, default=42)
args = argp.parse_args()

# =========================
# Reproducibility
# =========================
random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

# =========================
# Helpers
# =========================
def collect_images(root):
    root = Path(root)
    valid_ext = (".png", ".jpg", ".jpeg", ".bmp")
    return sorted([str(p) for p in root.rglob("*") if p.suffix.lower() in valid_ext])

def extract_label(path):
    return Path(path).name[:2]

def gray(img):
    if img is None:
        return None
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# =========================
# Segmentation
# =========================
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

# =========================
# SAFE normalization (FIX)
# =========================
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
# Load Dataset
# =========================
files = collect_images(args.train_dir)
if not files:
    raise SystemExit("No training data found")

classes = sorted({extract_label(f) for f in files})
cls_to_idx = {c: i for i, c in enumerate(classes)}

X_data, y_data = [], []

for fp in tqdm(files, desc="Preparing data"):
    img = cv2.imread(fp)
    if img is None:
        continue

    label = cls_to_idx[extract_label(fp)]
    g = gray(img)
    lines = find_text_lines(g)

    for y1, y2 in lines:
        segment = img[y1:y2, :]
        words = find_words(segment) or [segment]

        for w in words:
            chars = split_characters(w, args.min_width)
            chars = chars if chars else [w]

            for c in chars:
                ch = normalize_char(c, args.img_h, args.img_w)
                if ch is not None:     # 🔒 CRITICAL FIX
                    X_data.append(ch)
                    y_data.append(label)

X_data = np.array(X_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.int32)

# Shuffle
perm = np.random.permutation(len(X_data))
X_data, y_data = X_data[perm], y_data[perm]

y_cat = to_categorical(y_data, num_classes=len(classes))

# Split
split = max(1, int(args.val_ratio * len(X_data)))
X_val, y_val = X_data[:split], y_cat[:split]
X_tr, y_tr = X_data[split:], y_cat[split:]

# Class weights
freq = Counter(y_data.tolist())
weights = {i: len(y_data) / (len(freq) * freq[i]) for i in freq}

# =========================
# Model
# =========================
def make_model(shape, n_cls):
    inp = layers.Input(shape=shape)
    x = inp

    for f in [32, 64, 128, 256, 256]:
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    out = layers.Dense(n_cls, activation="softmax")(x)
    return models.Model(inp, out)

model = make_model((args.img_h, args.img_w, 3), len(classes))
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_list = [
    callbacks.ModelCheckpoint(args.model_out, monitor="val_accuracy", save_best_only=True),
    callbacks.ReduceLROnPlateau(patience=4, factor=0.5),
    callbacks.EarlyStopping(patience=12, restore_best_weights=True)
]

model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=args.epochs,
    batch_size=args.batch,
    shuffle=True,
    class_weight=weights,
    callbacks=callbacks_list
)

model.save(args.model_out)
with open(args.labels_out, "w") as f:
    json.dump(classes, f, indent=2)

print(" Training complete.")
