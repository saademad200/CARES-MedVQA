"""
eval_task2.py — Subtask-2 confidence/safety evaluation against val ground truth.

Subtask 2 is expert-rated (no automatic leaderboard metric), so these are *proxy*
diagnostics for the "confidence calibration" and "safety" criteria:
  • answer correctness (Task-1 answer vs val GT, normalised)
  • calibration: ECE, MCE, Brier, AUROC(confidence → correctness), reliability bins
  • selective prediction: risk–coverage AUC, selective accuracy at fixed coverage
  • safety hedge: hedge rate, hedge precision (errors caught), over-hedge rate

Reports TWO confidence sources:
  [in-sample]    confidence_score as written in the jsonl (= full-val per-aspect
                 reliability; this is the *deployment* setting for the hidden test).
  [cross-fitted] honest held-out estimate — each row's confidence comes from a
                 per-aspect reliability table fit on the OTHER K folds (split by
                 img_id), never its own fold. This is the number to report.

Usage: python eval_task2.py [--jsonl submission_task2.jsonl] [--bins 15] [--folds 5]
"""
import os
import json
import argparse
from collections import defaultdict

import numpy as np
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


def auroc(scores, labels):
    n_pos = sum(labels); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores); i = 0
    while i < len(order):
        j = i
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    sum_pos = sum(ranks[i] for i in range(len(scores)) if labels[i] == 1)
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def ece_bins(conf, correct, M=15):
    bins = [[] for _ in range(M)]
    for c, y in zip(conf, correct):
        bins[min(M - 1, int(c * M))].append((c, y))
    N = len(conf); ece = 0.0; mce = 0.0; rows = []
    for b in range(M):
        if not bins[b]:
            continue
        cs = [x[0] for x in bins[b]]; ys = [x[1] for x in bins[b]]
        conf_b = sum(cs) / len(cs); acc_b = sum(ys) / len(ys); n = len(ys)
        gap = abs(acc_b - conf_b)
        ece += gap * n / N; mce = max(mce, gap)
        rows.append({"lo": round(b / M, 3), "hi": round((b + 1) / M, 3),
                     "n": n, "conf": round(conf_b, 4), "acc": round(acc_b, 4)})
    return ece, mce, rows


def metrics(conf, correct, M, coverage, floor, abstain):
    N = len(conf)
    brier = sum((c - y) ** 2 for c, y in zip(conf, correct)) / N
    ece, mce, rel = ece_bins(conf, correct, M)
    au = auroc(conf, correct)
    order = sorted(range(N), key=lambda i: -conf[i])
    cum = 0; rc = []
    for k, i in enumerate(order, 1):
        cum += (1 - correct[i]); rc.append(cum / k)
    rc_auc = sum(rc) / len(rc); sel = 1 - rc[int(coverage * N) - 1]
    # hedge tiers from the (this-source) confidence vs the calibrated thresholds
    hedged = [i for i in range(N) if conf[i] < floor]
    unh = [i for i in range(N) if conf[i] >= floor]
    err_h = (sum(1 - correct[i] for i in hedged) / len(hedged)) if hedged else float("nan")
    err_u = (sum(1 - correct[i] for i in unh) / len(unh)) if unh else float("nan")
    over = (sum(1 for i in range(N) if correct[i] and conf[i] < floor) / sum(correct)) if sum(correct) else 0
    n_err = sum(1 - c for c in correct)
    err_capture = (sum(1 for i in hedged if not correct[i]) / n_err) if n_err else float("nan")
    conf_wrong = sum(1 for i in unh if not correct[i]) / N            # asserted-AND-wrong (clinical risk)
    assert_acc = (sum(correct[i] for i in unh) / len(unh)) if unh else float("nan")
    return {
        "ECE": round(ece, 4), "MCE": round(mce, 4), "Brier": round(brier, 4),
        "AUROC_conf_vs_correct": round(au, 4),
        "risk_coverage_AUC": round(rc_auc, 4), f"sel_acc@cov{coverage}": round(sel, 4),
        "hedge_rate": round(len(hedged) / N, 4),
        "err_rate_hedged(=hedge_precision)": round(err_h, 4), "err_rate_unhedged": round(err_u, 4),
        "error_capture(recall_of_errors)": round(err_capture, 4),
        "confidently_wrong_rate": round(conf_wrong, 4),
        "asserted_accuracy": round(assert_acc, 4),
        "over_hedge_rate": round(over, 4),
    }, rel


def crossfit_confidence(rows, correct, K, alpha=5.0):
    """Honest held-out confidence: per-(aspect, answer) reliability, shrunk toward the aspect
    mean (empirical Bayes), fit on the OTHER K folds (split by img_id) — never the row's own."""
    imgs = sorted({r["img_id"] for r in rows})
    fold_of = {iid: i % K for i, iid in enumerate(imgs)}
    asp = [get_required_aspects(r["question"]) for r in rows]
    ans = [norm(r["answer"]) for r in rows]
    foldr = [fold_of[r["img_id"]] for r in rows]
    out = [0.0] * len(rows)
    for f in range(K):
        a_acc = defaultdict(lambda: [0, 0]); c_acc = defaultdict(lambda: [0, 0]); G = [0, 0]
        for i in range(len(rows)):
            if foldr[i] == f:
                continue
            a_acc[asp[i]][0] += correct[i]; a_acc[asp[i]][1] += 1
            c_acc[(asp[i], ans[i])][0] += correct[i]; c_acc[(asp[i], ans[i])][1] += 1
            G[0] += correct[i]; G[1] += 1
        gmean = G[0] / G[1] if G[1] else 0.5
        for i in range(len(rows)):
            if foldr[i] != f:
                continue
            am = a_acc[asp[i]][0] / a_acc[asp[i]][1] if a_acc[asp[i]][1] else gmean
            c = c_acc.get((asp[i], ans[i]), [0, 0])
            out[i] = (c[0] + alpha * am) / (c[1] + alpha)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="submission_task2.jsonl")
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--coverage", type=float, default=0.8)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    try:
        ar = json.load(open("aspect_reliability.json"))
        floor, abstain, alpha = ar.get("_floor", 0.78), ar.get("_abstain", 0.60), ar.get("_alpha", 5.0)
    except FileNotFoundError:
        floor, abstain, alpha = 0.78, 0.60, 5.0

    rows = [json.loads(l) for l in open(args.jsonl)]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    gt = list(load_dataset("SimulaMet/Kvasir-VQA-test", split="validation")["answer"])

    conf_js = [float(r["confidence_score"]) for r in rows]
    correct = [int(norm(r["answer"]) == norm(gt[r["val_id"]])) for r in rows]
    grounded = sum(1 for r in rows if r["visual_explanation"])
    conf_xf = crossfit_confidence(rows, correct, args.folds, alpha)

    N = len(rows)
    print(json.dumps({
        "n": N, "answer_accuracy": round(sum(correct) / N, 4),
        "grounding_rate": round(grounded / N, 4),
        "hedge_thresholds": {"floor": floor, "abstain": abstain},
    }, indent=2))

    m_js, _ = metrics(conf_js, correct, args.bins, args.coverage, floor, abstain)
    m_xf, rel_xf = metrics(conf_xf, correct, args.bins, args.coverage, floor, abstain)
    print("\n[in-sample]    (jsonl confidence = full-val table; deployment setting)")
    print(json.dumps(m_js, indent=2))
    print(f"\n[cross-fitted] (honest held-out, {args.folds}-fold by img_id — REPORT THIS)")
    print(json.dumps(m_xf, indent=2))

    print("\nCross-fitted reliability diagram (bin: conf -> acc, n):")
    for r in rel_xf:
        print(f"  [{r['lo']:.2f}-{r['hi']:.2f}] conf={r['conf']:.3f} acc={r['acc']:.3f} "
              f"n={r['n']:<4} {'#' * int(r['acc'] * 20)}")

    # ── Interpretability proxies ────────────────────────────────────────────────
    # (1) confidence faithfulness = AUROC + ECE above. (2) explanation faithfulness:
    # fraction of explanations whose stated location/colour match the image's Task-1
    # answers (i.e. the explanation cites verifiable facts, not invented ones).
    interp = {"confidence_faithfulness_AUROC": m_xf["AUROC_conf_vs_correct"],
              "confidence_calibration_ECE": m_xf["ECE"]}
    try:
        t1 = json.load(open("predictions_1.json"))["predictions"]
        loc = {p["img_id"]: p["answer"] for p in t1 if get_required_aspects(p["question"]) == "location"}
        col = {p["img_id"]: p["answer"] for p in t1 if get_required_aspects(p["question"]) == "color"}
        loc_ok = loc_n = col_ok = col_n = 0
        for r in rows:
            e = r["textual_explanation"].lower()
            la = loc.get(r["img_id"], "")
            if la and la.lower() not in ("none", ""):
                toks = [t.strip().lower() for t in la.split(";") if t.strip()][:1]
                if toks and toks[0] in e:
                    loc_ok += 1
                loc_n += 1
            ca = col.get(r["img_id"], "")
            if ca and ca.lower() not in ("none", "") and "coloration" in e:
                cs = [c.strip().lower() for c in ca.split(";") if c.strip()]
                if cs and cs[0] in e:
                    col_ok += 1
                col_n += 1
        # mention-rate (coverage), NOT a faithfulness failure when low; colour-consistency is faithfulness.
        interp["explanation_location_mention_rate"] = round(loc_ok / loc_n, 4) if loc_n else None
        interp["explanation_colour_consistency"] = round(col_ok / col_n, 4) if col_n else None
    except FileNotFoundError:
        pass
    try:  # (3) grounding faithfulness: mask centroid vs Task-1 location, for grounded rows
        raws = [json.loads(l) for l in open("raw_cache.jsonl")]
        loc = {p["img_id"]: p["answer"] for p in json.load(open("predictions_1.json"))["predictions"]
               if get_required_aspects(p["question"]) == "location"}
        ok = tot = 0
        for r in raws:
            reg, la = r.get("region"), loc.get(r["img_id"], "")
            if reg and la and la.lower() not in ("none", ""):
                tot += 1
                if reg.lower() in {t.strip().lower() for t in la.split(";")}:
                    ok += 1
        interp["grounding_spatial_agreement"] = round(ok / tot, 4) if tot else None
    except FileNotFoundError:
        pass
    print("\n[interpretability]")
    print(json.dumps(interp, indent=2))

    json.dump({"in_sample": m_js, "cross_fitted": m_xf, "reliability_xf": rel_xf,
               "interpretability": interp, "accuracy": round(sum(correct) / N, 4), "n": N},
              open("eval_task2_report.json", "w"), indent=2)
    print("\n✅ Saved eval_task2_report.json")


if __name__ == "__main__":
    main()
