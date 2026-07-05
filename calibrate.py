"""
calibrate.py — estimate per-aspect reliability P(correct | question type) on a
labelled calibration set, and write aspect_reliability.json used by submission_task2.py
as the (calibrated) confidence_score.

Rationale: instance-level signals (explanation likelihood AUROC 0.59, answer
self-consistency 0.62) barely predict correctness here, but errors are strongly
aspect-structured — per-aspect reliability reaches AUROC ~0.78 (2-fold). This is a
training-free post-hoc calibrator: no model weights change; the table is a handful of
scalars fit on labelled data and applied to the unlabelled competition test.

Calibration set: by default the val jsonl + its GT answers (which serves as the
held-out calibration set for the hidden test). Reported thesis AUROC/ECE should use the
2-fold cross-fit in eval — see docs/Task2_Thesis_Implementation_Plan.md §6.4.

Usage: python calibrate.py [--jsonl submission_task2.jsonl] [--floor 0.78] [--abstain 0.60]
"""
import os, json, argparse
from collections import defaultdict

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from datasets import load_dataset


def get_required_aspects(question):
    q = str(question).lower()
    if "color" in q or "colour" in q: return "color"
    if "where" in q or "location" in q or "located" in q: return "location"
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="submission_task2.jsonl")
    # Interpretable, since confidence is a calibrated P(correct): assert >= floor,
    # mild hedge in [abstain, floor), strong hedge/abstain < abstain (more likely wrong than right).
    ap.add_argument("--floor", type=float, default=0.80, help="confidence below this -> mild hedge")
    ap.add_argument("--abstain", type=float, default=0.50, help="confidence below this -> strong hedge")
    ap.add_argument("--alpha", type=float, default=5.0,
                    help="empirical-Bayes shrinkage of per-(aspect,answer) cells toward the aspect mean")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.jsonl)]
    gt = list(load_dataset("SimulaMet/Kvasir-VQA-test", split="validation")["answer"])

    agg = defaultdict(lambda: [0, 0])     # per aspect
    cells = defaultdict(lambda: [0, 0])   # per (aspect, normalized answer)
    for r in rows:
        a = get_required_aspects(r["question"])
        y = int(norm(r["answer"]) == norm(gt[r["val_id"]]))
        agg[a][0] += y; agg[a][1] += 1
        cells[f"{a}|||{norm(r['answer'])}"][0] += y
        cells[f"{a}|||{norm(r['answer'])}"][1] += 1

    rel = {a: round(c / n, 4) for a, (c, n) in agg.items()}
    glob = round(sum(c for c, n in agg.values()) / sum(n for c, n in agg.values()), 4)
    # Confidence is the per-(aspect,answer) reliability, shrunk toward the aspect mean for
    # small cells (empirical Bayes). Cross-fitted AUROC 0.87 vs 0.77 for per-aspect alone.
    out = {"reliability": rel, "cells": {k: v for k, v in cells.items()},
           "_global": glob, "_alpha": args.alpha,
           "_floor": args.floor, "_abstain": args.abstain,
           "_n": sum(n for _, n in agg.values()), "_source": args.jsonl}
    json.dump(out, open("aspect_reliability.json", "w"), indent=2)

    print(f"Calibration set: {out['_n']} rows; global acc={glob}; cells={len(cells)}; alpha={args.alpha}")
    for a, p in sorted(rel.items(), key=lambda x: x[1]):
        tier = "ABSTAIN" if p < args.abstain else ("hedge" if p < args.floor else "assert")
        print(f"  {a:20s} aspect_reliability={p:.3f}  -> {tier}")
    print(f"\n✅ Wrote aspect_reliability.json (per-aspect+answer cells, floor={args.floor}, abstain={args.abstain})")


if __name__ == "__main__":
    main()
