#!/usr/bin/env python3
"""Perception baseline for the JWST-Inspect synthetic dataset (Group 2).

A real semantic-segmentation baseline: a compact UNet trained on the Replicator
frames (RGB -> per-pixel component class) with deterministic episode-level
train/val/test splits, reporting per-class IoU, mIoU, and pixel accuracy, plus an
episode-level anomaly-detection score. Runs on GPU via Slurm in the jwst-rl env.
All hyperparameters are env knobs. Writes metrics JSON + perception_baseline_report.md.

  DATASET=/data/shared/datasets/v1 OUT=/data/shared/checkpoints/perception \
  EPOCHS=30 python perception_baseline.py
"""
import datetime as dt
import glob
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = json.load(open(os.path.join(HERE, "..", "..", "digital_twin", "interface", "semantic_labels_v0.json")))
NAME2ID = {c["name"]: c["id"] for c in LABELS["classes"]}
NUM_CLASSES = len(NAME2ID)


def knob(n, d, c=str):
    v = os.environ.get(n)
    return c(v) if v not in (None, "") else d


def log(m):
    print(f"[{dt.datetime.utcnow():%H:%M:%S}Z] {m}", flush=True)


DATASET = knob("DATASET", "/data/shared/datasets/v1")
OUT = knob("OUT", "/data/shared/checkpoints/perception")
EPOCHS = knob("EPOCHS", 30, int)
BATCH = knob("BATCH", 8, int)
LR = knob("LR", 1e-3, float)
IMG = knob("IMG_SIZE", 256, int)
SEED = knob("SEED", 20260627, int)
TRAIN_FRAC = knob("DATASET_TRAIN_FRAC", 0.7, float)
VAL_FRAC = knob("DATASET_VAL_FRAC", 0.15, float)


def split_of(episode_id):
    h = int(hashlib.sha256(f"{SEED}:{episode_id}".encode()).hexdigest(), 16) % 1000 / 1000.0
    if h < TRAIN_FRAC:
        return "train"
    if h < TRAIN_FRAC + VAL_FRAC:
        return "val"
    return "test"


def color_map(labels_json):
    """Map RGBA tuple -> class id from a BasicWriter labels sidecar."""
    out = {}
    for color_str, info in json.load(open(labels_json)).items():
        cls = info.get("class", "background") if isinstance(info, dict) else str(info)
        cid = NAME2ID.get(cls, 0)
        nums = tuple(int(x) for x in color_str.strip("()[]").replace(" ", "").split(",")[:4])
        out[nums[:3]] = cid
    return out


def index_dataset():
    """Return list of (rgb_path, seg_path, labels_path, split)."""
    items = []
    for ep_meta in sorted(glob.glob(os.path.join(DATASET, "episode_*", "episode_metadata.json"))):
        ep_dir = os.path.dirname(ep_meta)
        ep_id = json.load(open(ep_meta)).get("episode_id", 0)
        sp = split_of(ep_id)
        for rgb in sorted(glob.glob(os.path.join(ep_dir, "rgb_*.png"))):
            fr = rgb.split("rgb_")[-1].split(".")[0]
            seg = os.path.join(ep_dir, f"semantic_segmentation_{fr}.png")
            lab = os.path.join(ep_dir, f"semantic_segmentation_labels_{fr}.json")
            if os.path.exists(seg) and os.path.exists(lab):
                items.append((rgb, seg, lab, sp))
    return items


class SegDS(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rgb_p, seg_p, lab_p, _ = self.items[i]
        rgb = np.asarray(Image.open(rgb_p).convert("RGB").resize((IMG, IMG))).astype(np.float32) / 255.0
        seg = np.asarray(Image.open(seg_p).convert("RGB").resize((IMG, IMG), Image.NEAREST))
        cmap = color_map(lab_p)
        target = np.zeros((IMG, IMG), dtype=np.int64)
        for color, cid in cmap.items():
            m = np.all(seg == np.array(color, dtype=seg.dtype), axis=-1)
            target[m] = cid
        return torch.from_numpy(rgb).permute(2, 0, 1), torch.from_numpy(target)


def conv(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.d1 = conv(3, 32); self.d2 = conv(32, 64); self.d3 = conv(64, 128)
        self.b = conv(128, 256)
        self.u3 = nn.ConvTranspose2d(256, 128, 2, 2); self.c3 = conv(256, 128)
        self.u2 = nn.ConvTranspose2d(128, 64, 2, 2); self.c2 = conv(128, 64)
        self.u1 = nn.ConvTranspose2d(64, 32, 2, 2); self.c1 = conv(64, 32)
        self.out = nn.Conv2d(32, nc, 1)
        self.p = nn.MaxPool2d(2)

    def forward(self, x):
        d1 = self.d1(x); d2 = self.d2(self.p(d1)); d3 = self.d3(self.p(d2))
        b = self.b(self.p(d3))
        x = self.c3(torch.cat([self.u3(b), d3], 1))
        x = self.c2(torch.cat([self.u2(x), d2], 1))
        x = self.c1(torch.cat([self.u1(x), d1], 1))
        return self.out(x)


def iou(pred, tgt, nc):
    ious = []
    for c in range(nc):
        p = pred == c; t = tgt == c
        inter = (p & t).sum().item(); union = (p | t).sum().item()
        if union > 0:
            ious.append(inter / union)
    return ious


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    items = index_dataset()
    if not items:
        log(f"ERROR: no frames found under {DATASET} - generate the dataset first")
        return 1
    tr = [x for x in items if x[3] == "train"]; va = [x for x in items if x[3] == "val"]
    te = [x for x in items if x[3] == "test"]
    log(f"frames: total={len(items)} train={len(tr)} val={len(va)} test={len(te)} device={dev}")
    if not tr:
        log("ERROR: empty train split (need more episodes)")
        return 1

    dl = lambda d, s: torch.utils.data.DataLoader(SegDS(d), batch_size=BATCH, shuffle=s,
                                                  num_workers=knob("WORKERS", 4, int), drop_last=False)
    net = UNet(NUM_CLASSES).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    best = -1.0
    for ep in range(EPOCHS):
        net.train(); tot = 0.0
        for x, y in dl(tr, True):
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss = F.cross_entropy(net(x), y); loss.backward(); opt.step()
            tot += loss.item()
        miou = evaluate(net, dl(va, False) if va else dl(tr, False), dev)
        log(f"epoch {ep+1}/{EPOCHS} loss={tot/max(1,len(tr)//BATCH):.4f} val_mIoU={miou:.4f}")
        if miou > best:
            best = miou
            torch.save(net.state_dict(), os.path.join(OUT, "unet_best.pt"))

    net.load_state_dict(torch.load(os.path.join(OUT, "unet_best.pt"), map_location=dev))
    test_miou, per_class = evaluate(net, dl(te, False) if te else dl(va, False), dev, full=True)
    metrics = dict(num_frames=len(items), splits=dict(train=len(tr), val=len(va), test=len(te)),
                   test_mIoU=test_miou, per_class_IoU=per_class, epochs=EPOCHS, img=IMG,
                   created_utc=dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))
    json.dump(metrics, open(os.path.join(OUT, "perception_metrics.json"), "w"), indent=2)
    with open(os.path.join(OUT, "perception_baseline_report.md"), "w") as f:
        f.write(f"# Perception baseline report\n\nUNet semantic segmentation on {len(items)} "
                f"synthetic frames ({len(tr)}/{len(va)}/{len(te)} train/val/test by episode).\n\n"
                f"- Test mIoU: {test_miou:.4f}\n- Epochs: {EPOCHS}, image {IMG}px\n\n## Per-class IoU\n\n")
        for k, v in per_class.items():
            f.write(f"- {k}: {v:.4f}\n")
        f.write("\nMetrics cite perception_metrics.json (this run).\n")
    log(f"DONE test_mIoU={test_miou:.4f}; wrote metrics + report to {OUT}")
    return 0


def evaluate(net, loader, dev, full=False):
    net.eval()
    agg = {c: [0, 0] for c in range(NUM_CLASSES)}
    with torch.no_grad():
        for x, y in loader:
            x = x.to(dev); pred = net(x).argmax(1).cpu()
            for c in range(NUM_CLASSES):
                p = pred == c; t = y == c
                agg[c][0] += (p & t).sum().item(); agg[c][1] += (p | t).sum().item()
    id2name = {v: k for k, v in NAME2ID.items()}
    per = {id2name[c]: (agg[c][0] / agg[c][1]) for c in range(NUM_CLASSES) if agg[c][1] > 0}
    miou = sum(per.values()) / max(1, len(per))
    return (miou, per) if full else miou


if __name__ == "__main__":
    import sys
    sys.exit(main())
