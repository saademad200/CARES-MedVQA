"""
exp_selfconsistency.py — hypothesis test: does answer self-consistency predict
correctness better than the explanation-likelihood confidence (AUROC 0.59)?

Samples the <MedVQA> answer T times (stochastic) per item, measures modal-answer
agreement, and reports AUROC of agreement vs. Task-1 answer correctness on a subset.
Reuses the already-loaded model from submission_task2 (import side-effects load it).

Usage: python exp_selfconsistency.py [N] [T]
"""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from collections import Counter

import torch
import submission_task2 as st
st.load_models()
model_hf, processor, device = st.model_hf, st.processor, st.device
val_dataset, task1_answers = st.val_dataset, st.task1_answers


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


def sample_answers(question, image, T):
    prompt = f"<MedVQA> Question: {question}"
    inp = processor(text=[prompt], images=[image], return_tensors="pt")
    inp = {k: v.to(device) for k, v in inp.items() if k != "labels"}
    if "pixel_values" in inp:
        inp["pixel_values"] = inp["pixel_values"].half()
    outs = []
    with torch.no_grad():
        for _ in range(T):
            g = model_hf.generate(**inp, max_new_tokens=64, do_sample=True,
                                  temperature=0.7, top_p=0.95)
            raw = processor.batch_decode(g, skip_special_tokens=False)[0]
            p = processor.post_process_generation(raw, task="<MedVQA>",
                                                 image_size=(image.width, image.height))
            outs.append(p.get("<MedVQA>", "").strip().lower())
    return outs


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    gt = list(val_dataset["answer"])
    sc, correct = [], []
    for idx in range(min(N, len(val_dataset))):
        ex = val_dataset[idx]
        outs = sample_answers(ex["question"], ex["image"].convert("RGB"), T)
        modal, cnt = Counter(outs).most_common(1)[0]
        sc.append(cnt / T)
        correct.append(int(norm(task1_answers.get(idx, "")) == norm(gt[idx])))
        if (idx + 1) % 25 == 0:
            print(f"  {idx+1}/{N} done", flush=True)
    acc = sum(correct) / len(correct)
    au = auroc(sc, correct)
    # accuracy split by consistency level
    hi = [correct[i] for i in range(len(sc)) if sc[i] >= 0.8]
    lo = [correct[i] for i in range(len(sc)) if sc[i] < 0.8]
    print(f"\nN={len(sc)} T={T}  accuracy={acc:.3f}")
    print(f"AUROC(self_consistency -> correct) = {au:.3f}   (vs explanation-likelihood 0.592)")
    print(f"acc | consistency>=0.8 : {sum(hi)/len(hi):.3f} (n={len(hi)})" if hi else "no hi")
    print(f"acc | consistency<0.8  : {sum(lo)/len(lo):.3f} (n={len(lo)})" if lo else "no lo")
