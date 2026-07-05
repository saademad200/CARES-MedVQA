"""
exp_retrieval_task2.py — does retrieval add anything to per-aspect reliability for
predicting Task-2 answer correctness? (CPU-only; uses raw_cache.jsonl + the FAISS index.)

Compares AUROC(correctness) of:
  - per-aspect reliability (current confidence)
  - visual_sim (top-1 cosine), coherence (1-dispersion)  [already cached]
  - neighbour-answer agreement with the predicted answer  [retrieval cross-check]
  - a cross-fitted logistic blend of all of them
If the blend ~= per-aspect alone, retrieval adds nothing as a confidence signal.
"""
import os, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from collections import defaultdict
from pathlib import Path

import numpy as np
import faiss
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

CACHE = Path.home() / ".cache" / "autoresearch_aqag"
ENTITY_KW = ["polyp", "ulcer", "oesophag", "esophag", "erythema", "snare", "forceps",
             "instrument", "cecum", "z-line", "paris", "gastro", "colon"]


def get_aspect(q):
    q = q.lower()
    if "color" in q or "colour" in q: return "color"
    if "where" in q or "located" in q or "location" in q: return "location"
    if "how many" in q or "count" in q:
        return "instrument_count" if ("instrument" in q or "instrumnet" in q) else "polyp_count"
    if "removed" in q or "removal" in q: return "polyp_removal"
    if "text" in q: return "text_presence"
    if "artefact" in q or "artifact" in q or "box" in q: return "artefact"
    if "polyp" in q: return "polyp_type" if "type" in q else "polyps"
    if "landmark" in q or "z-line" in q: return "anatomical_landmarks"
    if "instrument" in q or "instrumnet" in q: return "instruments"
    if "procedure" in q or "depicted" in q or "shown" in q: return "procedure"
    return "abnormalities"


def norm(s):
    return ";".join(sorted(t.strip().lower() for t in str(s).split(";") if t.strip()))


def kw(ans):
    a = ans.lower()
    for k in ENTITY_KW:
        if k in a:
            return k
    return a.strip()


rows = [json.loads(l) for l in open("raw_cache.jsonl")]
gt = list(load_dataset("SimulaMet/Kvasir-VQA-test", split="validation")["answer"])
rel = json.load(open("aspect_reliability.json"))["reliability"]
glob = json.load(open("aspect_reliability.json"))["_global"]

# Retrieval: train-only index + img_id-aligned val embeddings + train answer sentences
Xtr = np.load(CACHE / "train_vis_all.npy").astype("float32"); faiss.normalize_L2(Xtr)
idx = faiss.IndexFlatIP(Xtr.shape[1]); idx.add(Xtr)
train_answers = np.load(CACHE / "train_answers.npy", allow_pickle=True)
Xv = np.load(CACHE / "val_vis_all.npy").astype("float32"); faiss.normalize_L2(Xv)
vids = list(np.load(CACHE / "val_img_ids.npy", allow_pickle=True))
val_emb = {iid: Xv[i] for i, iid in enumerate(vids)}
K = 5

p_aspect, vsim, coh, support, correct, groups = [], [], [], [], [], []
for r in rows:
    asp = get_aspect(r["question"])
    p_aspect.append(rel.get(asp, glob))
    vsim.append(r.get("visual_sim") if r.get("visual_sim") is not None else 0.9)
    d = r.get("dispersion"); coh.append(1.0 - d if d is not None else 1.0)
    # neighbour-answer agreement with the predicted answer
    q = val_emb.get(r["img_id"])
    if q is not None:
        _, I = idx.search(q.reshape(1, -1), K)
        k = kw(r["answer"])
        support.append(np.mean([1.0 if k in str(train_answers[j]).lower() else 0.0 for j in I[0]]))
    else:
        support.append(0.0)
    correct.append(int(norm(r["answer"]) == norm(gt[r["val_id"]])))
    groups.append(r["img_id"])

y = np.array(correct)
feats = {"p_aspect": np.array(p_aspect), "visual_sim": np.array(vsim),
         "coherence": np.array(coh), "support": np.array(support)}
print(f"n={len(y)}  accuracy={y.mean():.3f}\n")
print("Single-signal AUROC(-> correct):")
for name, f in feats.items():
    print(f"  {name:12s}: {roc_auc_score(y, f):.4f}")

# Cross-fitted (5-fold by img_id) logistic blends
gid = {g: i % 5 for i, g in enumerate(sorted(set(groups)))}
fold = np.array([gid[g] for g in groups])


def xval_auroc(cols):
    X = np.column_stack([feats[c] for c in cols])
    oof = np.zeros(len(y))
    for f in range(5):
        tr, te = fold != f, fold == f
        if len(set(y[tr])) < 2:
            oof[te] = X[te].mean(1); continue
        m = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof)


print("\nCross-fitted logistic blends (5-fold by img_id):")
print(f"  p_aspect only            : {xval_auroc(['p_aspect']):.4f}")
print(f"  p_aspect + support       : {xval_auroc(['p_aspect','support']):.4f}")
print(f"  p_aspect + visual_sim    : {xval_auroc(['p_aspect','visual_sim']):.4f}")
print(f"  all 4 signals            : {xval_auroc(['p_aspect','visual_sim','coherence','support']):.4f}")
