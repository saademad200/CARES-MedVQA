"""
prepare_val.py — compute img_id-aligned AQAG visual embeddings for the Subtask-2
val set (SimulaMet/Kvasir-VQA-test). Encoder-only pass; mirrors autoresearch/prepare.py
exactly (same 4 frozen encoders, same preprocessing) so val vectors live in the SAME
4352-d space as the cached train index.

Why this exists: the cached `test_vis_all` is positionally keyed to
Kvasir-VQA-x1/test.jsonl, NOT to this val set — so it cannot be used here. See
docs/Task2_Thesis_Implementation_Plan.md §3.1.

Output (to ~/.cache/autoresearch_aqag/):
  val_vis_all.npy   (n_unique_images, 4352) float32
  val_img_ids.npy   (n_unique_images,) object   — row i ↔ img_id
"""
import gc
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from datasets import load_dataset

CACHE_DIR = Path.home() / ".cache" / "autoresearch_aqag"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Identical to autoresearch/prepare.py
ENCODER_CONFIGS = [
    ("dinov2",   "vit_large_patch14_dinov2.lvd142m", 1024, {"img_size": 224}),
    ("clip",     "vit_large_patch14_clip_224.laion2b", 1024, {}),
    ("siglip",   "vit_base_patch16_siglip_224.webli", 768, {}),
    ("convnext", "convnext_large.dinov3_lvd1689m", 1536, {}),
]
EVAL_TF = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


@torch.no_grad()
def extract(model_id, batch, extra_kwargs):
    import timm
    m = timm.create_model(model_id, pretrained=True, num_classes=0, **extra_kwargs).eval().to(DEVICE)
    out = []
    for i in range(0, len(batch), 64):
        x = batch[i:i + 64].to(DEVICE)
        with torch.cuda.amp.autocast():
            out.append(m(x).detach().float().cpu().numpy())
    del m
    torch.cuda.empty_cache()
    gc.collect()
    return np.concatenate(out)


if __name__ == "__main__":
    ds = load_dataset("SimulaMet/Kvasir-VQA-test", split="validation")
    # Unique images (first occurrence), preserving order.
    seen, images, img_ids = set(), [], []
    for ex in ds:
        iid = ex["img_id"]
        if iid not in seen:
            seen.add(iid)
            images.append(ex["image"].convert("RGB"))
            img_ids.append(iid)
    print(f"Unique val images: {len(images)} (from {len(ds)} rows)")

    batch = torch.stack([EVAL_TF(im) for im in images])
    feats = []
    for name, model_id, dim, kw in ENCODER_CONFIGS:
        print(f"Extracting {name} ({model_id})...")
        f = extract(model_id, batch, kw)
        assert f.shape[1] == dim, f"{name} dim {f.shape[1]} != {dim}"
        feats.append(f)

    X_val = np.concatenate(feats, axis=1).astype("float32")
    np.save(CACHE_DIR / "val_vis_all.npy", X_val)
    np.save(CACHE_DIR / "val_img_ids.npy", np.array(img_ids, dtype=object))
    print(f"Saved val_vis_all {X_val.shape} and val_img_ids ({len(img_ids)}).")
