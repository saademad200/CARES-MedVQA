# GutCheck — MedVQA-GI 2026 Subtask 2 (Multimodal Explainability & Safety)

## Method (training-free reliability layer over a frozen VLM)

Backbone: **Florence-2-base** + LoRA adapter `peeache/FL2_VQA_MIX_128_256`, used **frozen** —
no fine-tuning or retraining. All contributions are inference-time.

Per test case:

1. **Answer (Subtask 1).** The validated concise answer is taken from our Subtask-1 output
   (`predictions_1.json`).

2. **Visual grounding.** For answers naming a localizable entity (polyp, instrument, landmark,
   oesophagitis, …) we run `<REFERRING_EXPRESSION_SEGMENTATION>`, decode the polygon with the
   official `processor.post_process_generation`, draw an overlay, and emit it as a
   `segmentation_mask` in `visual_explanation` with a description tying it to the text. Non-
   localizable answers (Yes/No, counts, colours, procedures) are not grounded.

3. **Textual explanation.** Built around the validated answer (`answer_to_clause`), enriched with
   the **same image's other Subtask-1 aspect answers** (location, colour, size — zero extra model
   calls, consistent by construction), linked to the grounded region, and finished with the
   `<MedVQA_EXPLAIN>` narrative only when it is *consistent* with the answer and *adds new content*
   (contradiction + redundancy filters).

4. **Calibrated confidence (`confidence_score`).** We found that instance-level signals barely
   predict correctness on this frozen model (explanation likelihood AUROC 0.59; self-consistency
   0.62) and retrieval is *anti*-correlated (visual_sim AUROC 0.45) — errors are structured by
   **(aspect, answer value)**. So confidence is the **per-(aspect, answer) reliability**
   `P(correct | aspect, answer)` with empirical-Bayes shrinkage toward the aspect mean
   (`calibrate.py` → `aspect_reliability.json`). Cross-fitted (5-fold by `img_id`) it reaches
   **AUROC 0.87** (vs 0.77 per-aspect-only), ECE 0.025. No model weights are changed.

5. **Safety policy.** Because confidence is a calibrated `P(correct)`, the hedge thresholds are
   interpretable: `assert` (≥ 0.80), `mild hedge` (0.50–0.80), `strong/abstain` (< 0.50 — the
   model is more likely wrong than right). Hedged cases are **~9× more likely to be wrong** than
   asserted ones (0.47 vs 0.05 error rate), with a 0.15 over-hedge rate.

A FAISS retrieval layer (train-only index over a frozen DINOv2+CLIP+SigLIP+ConvNeXt ensemble,
`img_id`-aligned val embeddings, verified 0 leakage) is included; on this in-distribution set its
signals are weak and it is retained for ablation rather than driving the final confidence.

## Reproduction

```bash
# 1. (once) compute img_id-aligned val embeddings for the retrieval ablation
python prepare_val.py
# 2. build the per-aspect reliability calibrator from a labelled set
python calibrate.py
# 3. generate the submission (needs predictions_1.json in this directory)
python submission_task2.py --stage both       # GPU: infer + assemble (default); caches raw_cache.jsonl
#    iterate the explanation/confidence/hedge logic with NO GPU (~3s), reusing cached model outputs:
python submission_task2.py --stage assemble   # CPU-only, re-reads raw_cache.jsonl
# 4. evaluate calibration / safety against val GT (in-sample + 5-fold cross-fitted)
python eval_task2.py
# 5. (optional) reliability-diagram figure
python make_figures.py
```

Outputs: `submission_task2.jsonl` (one object per `val_id`, official schema) + `visuals/`.
`predictions_2.json` is an internal record only — Subtask 2 is expert-rated (no automatic
leaderboard metric), so no fidelity/FBD-style scores are emitted.

## Dependencies
`transformers`, `peft`, `torch`, `faiss-cpu`, `timm`, `numpy`, `Pillow`, `datasets`, `tqdm`.
