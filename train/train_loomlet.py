#!/usr/bin/env python3
"""Train the Loomlet scrap network and write the files the browser loads.

The twelve gallery lots and the upload fixtures are fictional patterns from
this script. They are not photographs of real cloth.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mirror_forward import forward as mirror_forward  # noqa: E402

FAMILIES = [
    "Harbor Stripe",
    "Moss Grid",
    "Copper Dot",
    "Night Speckle",
    "Tide Wave",
    "Cedar Chevron",
]

GALLERY = [
    ("HS-014", 0, "Pier-14", "Salt-ink", 1014),
    ("HS-027", 0, "Pier-27", "Fog lantern", 1027),
    ("MG-008", 1, "Fen-08", "Fen moss", 2008),
    ("MG-033", 1, "Fen-33", "Glass net", 2033),
    ("CD-002", 2, "Kiln-02", "Kiln copper", 3002),
    ("CD-019", 2, "Kiln-19", "Clay spark", 3019),
    ("NS-041", 3, "Lamp-41", "Lamp soot", 4041),
    ("NS-056", 3, "Lamp-56", "Moth dust", 4056),
    ("TW-011", 4, "Quay-11", "Quay teal", 5011),
    ("TW-028", 4, "Quay-28", "Night tide", 5028),
    ("CC-006", 5, "Loom-06", "Cedar bark", 6006),
    ("CC-022", 5, "Loom-22", "Shuttle red", 6022),
]


def two_colors(rng):
    for _ in range(30):
        a = rng.uniform(0.05, 0.95, size=3).astype(np.float32)
        b = rng.uniform(0.05, 0.95, size=3).astype(np.float32)
        if np.max(np.abs(a - b)) >= 0.35 and np.linalg.norm(a - b) >= 0.5:
            return a, b
    return (
        np.array([0.12, 0.16, 0.22], np.float32),
        np.array([0.9, 0.74, 0.28], np.float32),
    )


def motif(family, rng, size=32):
    ys, xs = np.mgrid[0:size, 0:size]
    if family == 0:
        period = 8
        thick = 2
        phase = int(rng.integers(0, period))
        mask = ((ys + phase) % period) < thick
    elif family == 1:
        px = 8
        py = 8
        thick = 1
        mask = ((xs % px) < thick) | ((ys % py) < thick)
    elif family == 2:
        period = 8
        radius = float(rng.uniform(1.8, 2.4))
        cx = (xs % period) - (period / 2.0)
        cy = (ys % period) - (period / 2.0)
        mask = (cx * cx + cy * cy) <= radius * radius
    elif family == 3:
        cell = 4
        density = float(rng.uniform(0.12, 0.22))
        salt = int(rng.integers(1, 9000))
        gx = (xs // cell).astype(np.int64)
        gy = (ys // cell).astype(np.int64)
        n = (gx * 374761393 + gy * 668265263 + salt * 1442695041) & 0xFFFFFFFF
        n = (n ^ (n >> 13)) * 1274126177 & 0xFFFFFFFF
        unit = (n & 0xFFFFFF) / 16777215.0
        mask = unit < density
    elif family == 4:
        px = int(rng.choice([8, 16]))
        py = int(rng.choice([16, 32]))
        amp = float(rng.uniform(1.0, 1.5))
        phase = float(rng.uniform(0, 6.28))
        wave = np.sin(2 * np.pi * xs / px + amp * np.sin(2 * np.pi * ys / py) + phase)
        mask = wave > 0.65
    else:
        period = 8
        fold = np.abs((xs % (2 * period)) - period)
        band = (ys + fold) % period
        mask = band < 2
    return mask.astype(np.float32)


def render(family, seed, size=32):
    rng = np.random.default_rng(seed)
    mask = motif(family, rng, size)
    dye_a, dye_b = two_colors(rng)
    shift_y = int(rng.integers(0, size))
    shift_x = int(rng.integers(0, size))
    mask = np.roll(np.roll(mask, shift_y, 0), shift_x, 1)
    image = dye_a * (1.0 - mask)[..., None] + dye_b * mask[..., None]
    image = np.clip(image + rng.normal(0, 0.015, image.shape).astype(np.float32), 0, 1)
    return image.astype(np.float32), mask.astype(np.float32)


def to_u8(image):
    return np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)


def from_u8(u8):
    return u8.astype(np.float32) / 255.0


def texture_std(image):
    worst = 0.0
    for channel in range(3):
        worst = max(worst, float(image[..., channel].std()))
    return worst


class Net(nn.Module):
    """Two networks that do not share weights.

    The classifier names the family. The segmenter lifts the thin motif.
    Separate weights keep those losses from fighting.
    """

    def __init__(self):
        super().__init__()
        self.cls1 = nn.Conv2d(3, 16, 3, padding=1)
        self.cls2 = nn.Conv2d(16, 24, 3, stride=2, padding=1)
        self.cls3 = nn.Conv2d(24, 32, 3, stride=2, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 48)
        self.fc2 = nn.Linear(48, 6)
        self.seg1 = nn.Conv2d(3, 16, 3, padding=1)
        self.seg2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.up = nn.Conv2d(48, 16, 3, padding=1)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        c = F.relu(self.cls1(x))
        c = F.relu(self.cls2(c))
        c = F.relu(self.cls3(c))
        emb = F.relu(self.fc1(torch.flatten(c, 1)))
        logits = self.fc2(emb)
        h1 = F.relu(self.seg1(x))
        h2 = F.relu(self.seg2(h1))
        d = F.interpolate(h2, scale_factor=2, mode="nearest")
        d = F.relu(self.up(torch.cat([d, h1], dim=1)))
        raw = self.head(d)
        return logits, emb, h1, h2, torch.sigmoid(raw), raw


def make_batch(n, rng):
    images = np.empty((n, 3, 32, 32), np.float32)
    masks = np.empty((n, 1, 32, 32), np.float32)
    labels = np.empty((n,), np.int64)
    for i in range(n):
        family = int(rng.integers(0, 6))
        seed = int(rng.integers(1, 2_000_000_000))
        image, mask = render(family, seed)
        u8 = to_u8(image)
        images[i] = np.transpose(from_u8(u8), (2, 0, 1))
        masks[i, 0] = mask
        labels[i] = family
    return images, masks, labels


def recolor(images, masks):
    """Same motif, new dyes. The embedding should ignore dye and keep structure."""
    n = images.shape[0]
    dye_a = torch.rand(n, 3, 1, 1)
    dye_b = torch.rand(n, 3, 1, 1)
    return (dye_a * (1.0 - masks) + dye_b * masks).clamp(0, 1)


def cosine_margin(emb, labels):
    """Push the worst same-family pairs up and the worst different-family pairs down."""
    z = F.normalize(emb, dim=1)
    sim = z @ z.transpose(0, 1)
    eye = torch.eye(z.shape[0], dtype=torch.bool)
    same = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye
    diff = labels.unsqueeze(0) != labels.unsqueeze(1)
    same_vals = sim[same]
    diff_vals = sim[diff]
    hard_same = torch.topk(same_vals, k=min(48, same_vals.numel()), largest=False).values
    hard_diff = torch.topk(diff_vals, k=min(64, diff_vals.numel()), largest=True).values
    return torch.relu(0.65 - hard_same).mean() + torch.relu(hard_diff - 0.2).mean()


def embedding_gap(emb, labels):
    """Robust gap: 10th percentile of same-family cosine minus 90th of different-family."""
    z = F.normalize(emb, dim=1)
    sim = z @ z.transpose(0, 1)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(sim.shape[0], dtype=torch.bool)
    same_vals = sim[same & ~eye]
    diff_vals = sim[~same]
    if same_vals.numel() == 0 or diff_vals.numel() == 0:
        return -1.0
    low = torch.quantile(same_vals.float(), 0.10)
    high = torch.quantile(diff_vals.float(), 0.90)
    return float(low - high)


def evaluate(model, n, seed):
    rng = np.random.default_rng(seed)
    images, masks, labels = make_batch(n, rng)
    x = torch.from_numpy(images)
    y = torch.from_numpy(labels)
    m = torch.from_numpy(masks)
    model.eval()
    with torch.no_grad():
        logits, emb, _, _, _, raw = model(x)
        pred = logits.argmax(1)
        acc = float((pred == y).float().mean())
        bce = float(F.binary_cross_entropy_with_logits(raw, m))
        pix = float(((raw > 0) == (m > 0.5)).float().mean())
        gap = embedding_gap(emb, y)
    return acc, bce, pix, gap


def tensor_nchw(image_hwc):
    return torch.from_numpy(np.transpose(image_hwc, (2, 0, 1))[None])


def softmax_np(logits):
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def weights_from_model(model):
    sd = model.state_dict()

    def w(name):
        return sd[name].detach().cpu().float().numpy()

    return {
        "cls1_w": w("cls1.weight"),
        "cls1_b": w("cls1.bias"),
        "cls2_w": w("cls2.weight"),
        "cls2_b": w("cls2.bias"),
        "cls3_w": w("cls3.weight"),
        "cls3_b": w("cls3.bias"),
        "fc1_w": w("fc1.weight"),
        "fc1_b": w("fc1.bias"),
        "fc2_w": w("fc2.weight"),
        "fc2_b": w("fc2.bias"),
        "seg1_w": w("seg1.weight"),
        "seg1_b": w("seg1.bias"),
        "seg2_w": w("seg2.weight"),
        "seg2_b": w("seg2.bias"),
        "up_w": w("up.weight"),
        "up_b": w("up.bias"),
        "head_w": w("head.weight"),
        "head_b": w("head.bias"),
    }


def flat(array):
    return [float(v) for v in np.asarray(array, np.float32).reshape(-1)]


def pack_layer(array_w, array_b):
    return {"w": flat(array_w), "wShape": list(array_w.shape), "b": flat(array_b)}


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def class_stats(model):
    rng = np.random.default_rng(4242)
    feat_sum = np.zeros((6, 32 * 16 * 16), np.float64)
    dye_sum = np.zeros((6, 2, 3), np.float64)
    dye_n = np.zeros((6, 2), np.float64)
    count = np.zeros(6, np.float64)
    model.eval()
    with torch.no_grad():
        for family in range(6):
            for _ in range(64):
                seed = int(rng.integers(1, 2_000_000_000))
                image, mask = render(family, seed)
                rt = from_u8(to_u8(image))
                _logits, _emb, _h1, h2, _, _ = model(tensor_nchw(rt))
                feat_sum[family] += h2[0].detach().cpu().numpy().reshape(-1)
                count[family] += 1
                for side, sel in ((0, mask < 0.5), (1, mask >= 0.5)):
                    if np.any(sel):
                        dye_sum[family, side] += rt[sel].mean(axis=0)
                        dye_n[family, side] += 1
    means = (feat_sum / count[:, None]).astype(np.float32)
    dyes = (dye_sum / np.maximum(dye_n[:, :, None], 1)).astype(np.float32)
    return means, dyes


def save_png(path, u8, size=None):
    image = Image.fromarray(u8, "RGB")
    if size:
        image = image.resize(size, Image.Resampling.NEAREST)
    image.save(path)


def main():
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 0))
    torch.manual_seed(20261011)
    rng = np.random.default_rng(20261011)
    model = Net()
    cls_params = [p for n, p in model.named_parameters() if n.startswith(("cls", "fc"))]
    seg_params = [p for n, p in model.named_parameters() if n.startswith(("seg", "up", "head"))]
    if len(cls_params) + len(seg_params) != len(list(model.parameters())):
        raise SystemExit("a parameter is missing from the optimizer groups")
    opt = torch.optim.Adam(
        [
            {"params": cls_params, "lr": 2e-3},
            {"params": seg_params, "lr": 1e-2},
        ]
    )
    best_state = None
    best_key = (-1.0, -1.0, -1.0)
    for epoch in range(1, 41):
        model.train()
        loss_sum = 0.0
        batches = 20
        for _ in range(batches):
            images, masks, labels = make_batch(128, rng)
            x = torch.from_numpy(images)
            y = torch.from_numpy(labels)
            m = torch.from_numpy(masks)
            opt.zero_grad()
            logits, emb, _, _, _, raw = model(x)
            _, emb_dye, _, _, _, _ = model(recolor(x, m))
            loss = (
                F.cross_entropy(logits, y)
                + F.binary_cross_entropy_with_logits(raw, m, pos_weight=m.new_tensor([3.0]))
                + (1.0 - F.cosine_similarity(emb, emb_dye, dim=1)).mean()
                + cosine_margin(emb, y)
            )
            loss.backward()
            # Clip each network on its own. A shared clip lets the classifier
            # swallow the step and the thin-motif head stays at the background rate.
            nn.utils.clip_grad_norm_(cls_params, 1.0)
            nn.utils.clip_grad_norm_(seg_params, 1.0)
            opt.step()
            loss_sum += float(loss.detach())
        acc, bce, pix, gap = evaluate(model, 600, 50_000 + epoch)
        print(
            f"epoch {epoch} loss {loss_sum / batches:.4f} acc {acc:.4f} bce {bce:.4f} pix {pix:.4f} gap {gap:.4f}",
            flush=True,
        )
        passed = int(acc >= 0.97 and pix >= 0.84)
        key = (passed, pix, acc)
        if key > best_key:
            best_key = key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    print(f"best val passed {best_key[0]} pix {best_key[1]:.4f} acc {best_key[2]:.4f}", flush=True)
    model.load_state_dict(best_state)
    model.eval()

    hold_acc, hold_bce, hold_pix, _hold_gap = evaluate(model, 720, 99991)
    print(f"holdout acc {hold_acc:.4f} bce {hold_bce:.4f} pix {hold_pix:.4f}", flush=True)
    if hold_acc < 0.97 or hold_pix < 0.84:
        raise SystemExit(f"holdout acc {hold_acc:.4f} pix {hold_pix:.4f} is too low")

    weights = weights_from_model(model)
    # One random tensor must match the NumPy loops before we trust the export.
    probe = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        t_logits, t_emb, _, _, t_mask, _ = model(probe)
    m_logits, m_emb, m_mask = mirror_forward(weights, probe[0].numpy().astype(np.float32))
    logit_err = float(np.max(np.abs(m_logits - t_logits[0].numpy())))
    mask_err = float(np.max(np.abs(m_mask - t_mask[0, 0].numpy())))
    emb_err = float(np.max(np.abs(m_emb - t_emb[0].numpy())))
    print(f"mirror err logits {logit_err:.3e} emb {emb_err:.3e} mask {mask_err:.3e}", flush=True)
    if max(logit_err, mask_err, emb_err) > 1e-4:
        raise SystemExit("NumPy mirror does not match PyTorch")

    means, dyes = class_stats(model)
    basis_rng = np.random.default_rng(7)
    basis = basis_rng.normal(size=(8, 32)).astype(np.float32)
    basis /= np.linalg.norm(basis, axis=1, keepdims=True) + 1e-8

    sample_dir = ROOT / "public" / "samples"
    pixel_dir = sample_dir / "pixels"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    fixture_dir = ROOT / "tests" / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    gallery_emb = []
    gallery_logits = []
    gallery_probs = []
    gallery_meta = []
    sheet = []
    min_conf = 1.0
    min_texture = 99.0
    for code, family, lot, nickname, seed in GALLERY:
        image, mask = render(family, seed)
        u8 = to_u8(image)
        rt = from_u8(u8)
        std = texture_std(rt)
        min_texture = min(min_texture, std)
        with torch.no_grad():
            logits, emb, _, _, pred_mask, _ = model(tensor_nchw(rt))
        probs = softmax_np(logits[0].numpy())
        pred = int(probs.argmax())
        conf = float(probs[family])
        min_conf = min(min_conf, conf)
        if pred != family or conf < 0.70:
            raise SystemExit(f"{code} predicted {pred} conf {conf:.3f}")
        m_logits, m_emb, m_mask = mirror_forward(weights, np.transpose(rt, (2, 0, 1)))
        if np.max(np.abs(m_logits - logits[0].numpy())) > 1e-4:
            raise SystemExit(f"mirror mismatch on {code}")
        save_png(sample_dir / f"{code}.png", u8, (256, 256))
        (pixel_dir / f"{code}.bin").write_bytes(u8.tobytes())
        # Display PNG must invert back to the same 32×32 bytes.
        big = Image.open(sample_dir / f"{code}.png").convert("RGB")
        small = np.asarray(big.resize((32, 32), Image.Resampling.NEAREST))
        if not np.array_equal(small, u8):
            raise SystemExit(f"display PNG is not a nearest enlargement: {code}")
        gallery_emb.append(emb[0].numpy().astype(np.float32))
        gallery_logits.append(logits[0].numpy().astype(np.float32))
        gallery_probs.append(probs.astype(np.float32))
        gallery_meta.append(
            {
                "id": code,
                "family": FAMILIES[family],
                "familyIndex": family,
                "lot": lot,
                "nickname": nickname,
                "seed": seed,
                "image": f"/samples/{code}.png",
                "pixels": f"/samples/pixels/{code}.bin",
                "confidence": round(conf, 4),
            }
        )
        recon = np.where(pred_mask[0, 0].numpy()[..., None] >= 0.5, rt.mean(axis=(0, 1)), rt.mean(axis=(0, 1)))
        del recon
        mask_u8 = (np.clip(pred_mask[0, 0].numpy(), 0, 1) * 255).astype(np.uint8)
        sheet.append((u8, mask_u8))
        if code == "HS-014":
            reference = {
                "id": code,
                "logits": [float(v) for v in m_logits],
                "embedding": [float(v) for v in m_emb],
                "mask": [float(v) for v in m_mask.reshape(-1)],
            }

    # Pairwise cosine. Family scores separate dye changes that the raw embedding mixes together.
    def gap_of(vectors):
        same_vals = []
        diff_vals = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                value = cosine(vectors[i], vectors[j])
                if gallery_meta[i]["familyIndex"] == gallery_meta[j]["familyIndex"]:
                    same_vals.append(value)
                else:
                    diff_vals.append(value)
        return float(min(same_vals)), float(max(diff_vals))

    emb_stack = np.stack(gallery_emb)
    logit_stack = np.stack(gallery_logits)
    # A shared bias makes every logit vector point the same way. Remove each vector's mean.
    logit_centered = logit_stack - logit_stack.mean(axis=1, keepdims=True)
    spaces = {
        "embedding": emb_stack,
        "centered": emb_stack - emb_stack.mean(axis=0),
        "logits": logit_centered,
        "softmax": np.stack(gallery_probs),
    }
    compare_on = None
    min_same = max_diff = threshold = None
    # Prefer the embedding when it separates. Family probabilities are the fallback.
    for name in ("embedding", "centered", "logits", "softmax"):
        same_min, diff_max = gap_of(spaces[name])
        print(f"cosine {name} same>={same_min:.4f} diff<={diff_max:.4f}", flush=True)
        if compare_on is None and same_min > diff_max + 0.02:
            compare_on = name
            min_same = same_min
            max_diff = diff_max
            threshold = (same_min + diff_max) / 2.0
            compare_vecs = spaces[name]
    if compare_on is None:
        np.savez(
            "/tmp/loomlet-gallery.npz",
            emb=emb_stack,
            logits=logit_stack,
            probs=np.stack(gallery_probs),
            family=np.array([row["familyIndex"] for row in gallery_meta], np.int64),
        )
        raise SystemExit("gallery readings overlap; compare threshold would be a guess")
    print(f"compare on {compare_on} threshold {threshold:.4f}", flush=True)

    # Explicit pairs used by the acceptance checks.
    def pair(a, b):
        ia = next(i for i, row in enumerate(gallery_meta) if row["id"] == a)
        ib = next(i for i, row in enumerate(gallery_meta) if row["id"] == b)
        return cosine(compare_vecs[ia], compare_vecs[ib])

    if pair("HS-014", "HS-027") < threshold:
        raise SystemExit("HS-014 and HS-027 would not count as the same family")
    if pair("HS-014", "MG-008") >= threshold:
        raise SystemExit("HS-014 and MG-008 would not count as different families")

    def write_fixture(path, family, seed, kind):
        image, _mask = render(family, seed)
        u8 = to_u8(image)
        if kind == "png":
            # Keep the native 32×32 grid so a browser upload is not a second resample.
            save_png(path, u8)
            opened = Image.open(path).convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
        else:
            Image.fromarray(u8, "RGB").resize((256, 256), Image.Resampling.NEAREST).save(path, quality=95)
            opened = Image.open(path).convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
        rt = from_u8(np.asarray(opened))
        with torch.no_grad():
            logits, _, _, _, _, _ = model(tensor_nchw(rt))
        probs = softmax_np(logits[0].numpy())
        pred = int(probs.argmax())
        conf = float(probs[family])
        print(f"fixture {path.name} pred {FAMILIES[pred]} conf {conf:.3f} texture {texture_std(rt):.3f}", flush=True)
        if pred != family or conf < 0.7:
            raise SystemExit(f"fixture {path.name} is not reliable enough")
        return conf

    write_fixture(fixture_dir / "tide-wave.png", 4, 5294, "png")
    write_fixture(fixture_dir / "moss-grid.jpg", 1, 2099, "jpg")
    flat = np.full((64, 64, 3), 180, np.uint8)
    Image.fromarray(flat, "RGB").save(fixture_dir / "flat-gray.png")
    (fixture_dir / "notes.txt").write_text("This is not an image.\n", encoding="utf-8")
    broken = bytes.fromhex("89504e470d0a1a0a0000000d49484452")
    (fixture_dir / "broken.png").write_bytes(broken)

    if min_texture < 0.08:
        raise SystemExit(f"a gallery lot is too flat ({min_texture:.3f})")

    model_dir = ROOT / "public" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "classes": FAMILIES,
        "sameFamilyCosineMin": threshold,
        "compareOn": compare_on,
        "embeddingCenter": [float(v) for v in emb_stack.mean(axis=0)] if compare_on == "centered" else None,
        "cls1": pack_layer(weights["cls1_w"], weights["cls1_b"]),
        "cls2": pack_layer(weights["cls2_w"], weights["cls2_b"]),
        "cls3": pack_layer(weights["cls3_w"], weights["cls3_b"]),
        "fc1": pack_layer(weights["fc1_w"], weights["fc1_b"]),
        "fc2": pack_layer(weights["fc2_w"], weights["fc2_b"]),
        "seg1": pack_layer(weights["seg1_w"], weights["seg1_b"]),
        "seg2": pack_layer(weights["seg2_w"], weights["seg2_b"]),
        "up": pack_layer(weights["up_w"], weights["up_b"]),
        "head": pack_layer(weights["head_w"], weights["head_b"]),
        "classH2Means": [[float(v) for v in row] for row in means],
        "seedBasis": [[float(v) for v in row] for row in basis],
        "classDyes": [[[float(v) for v in color] for color in row] for row in dyes],
    }
    (model_dir / "loomlet-weights.json").write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    (model_dir / "reference.json").write_text(json.dumps(reference), encoding="utf-8")
    metrics = {
        "holdoutAccuracy": hold_acc,
        "holdoutCount": 720,
        "holdoutMaskBce": hold_bce,
        "holdoutPixelAccuracy": hold_pix,
        "galleryMinConfidence": min_conf,
        "galleryMinTextureStd": min_texture,
        "sameFamilyCosineMin": threshold,
        "compareOn": compare_on,
        "samePairMin": min_same,
        "diffPairMax": max_diff,
        "families": FAMILIES,
        "note": "Scores use held-out synthetic images from the same fictional pattern generator as the sample lots. They are not measurements of real fabric.",
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (sample_dir / "manifest.json").write_text(json.dumps({"samples": gallery_meta}, indent=2), encoding="utf-8")

    # Contact sheet for a visual check: input beside the decoded motif.
    tile = 64
    canvas = Image.new("RGB", (tile * 2, tile * len(sheet)), (239, 231, 218))
    for index, (rgb, mask_u8) in enumerate(sheet):
        rgb_im = Image.fromarray(rgb, "RGB").resize((tile, tile), Image.Resampling.NEAREST)
        gray = Image.fromarray(mask_u8, "L").resize((tile, tile), Image.Resampling.NEAREST).convert("RGB")
        canvas.paste(rgb_im, (0, index * tile))
        canvas.paste(gray, (tile, index * tile))
    sheet_path = Path("/tmp/loomlet-preview.png")
    canvas.save(sheet_path)
    print(
        "LOOMLET_TRAIN",
        json.dumps(
            {
                "holdoutAccuracy": round(hold_acc, 4),
                "galleryMinConfidence": round(min_conf, 4),
                "threshold": round(threshold, 4),
                "minSame": round(min_same, 4),
                "maxDiff": round(max_diff, 4),
                "sheet": str(sheet_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
