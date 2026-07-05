# ImageCLEFmedical 2026 MEDVQA-GI — CEUR Working Notes

This folder contains the CEUR-WS working notes paper for our
**CARES** team submission to ImageCLEFmedical 2026 MEDVQA-GI:

- `working_notes.pdf` — **compiled PDF (8 pages, 1-column CEURART)**
- `working_notes.tex` — source
- `ceurart.cls`, `ccicons.sty` (+ font files) — class + dependencies
  (so anyone can recompile locally without further downloads)

**Authors**

1. Syed Saad Hasan Emad (syed.saad.31916@khi.iba.edu.pk)
2. Muhammad Atif Tahir (atiftahir@iba.edu.pk)
3. Itbaan Safwan

**Affiliation**: School of Mathematics and Computer Science,
Institute of Business Administration (IBA), Karachi, Pakistan

## What's in the paper

1. **Method.** Training-free reliability layer around a frozen
   Florence-2-base + `peeache/FL2_VQA_MIX_128_256` LoRA. Subtask 1
   uses a deterministic concise normalizer; Subtask 2 adds calibrated
   confidence + self-probed faithful explanations + segmentation
   overlays + a safety policy.
2. **Calibration result.** Per-(aspect, answer) reliability reaches
   **AUROC 0.87** for predicting answer correctness, vs ~0.60 for
   max-softmax / temperature scaling and 0.62 for self-consistency.
3. **Conformal upgrade.** Distribution-free selective-risk control
   (exact Clopper–Pearson UCB, δ = 0.1) — 55% coverage at a verified
   ≤ 5% error rate; guarantee holds in all folds.
4. **Faithfulness frontier (three independent metrics).** Template
   dominates LLM-narrated variants on structured claim support, NLI
   entailment, and an LLM-as-judge. The keyword gate gives false
   assurance.
5. **Negative results.** Instance-level uncertainty (MSP,
   self-consistency, generation likelihood), visual retrieval, and
   segmentation grounding all fail to predict correctness — the
   "base-rate dominance" phenomenon.

## Compiling

The paper uses the official CEUR-WS class file `ceurart.cls`.
Download the class + style files from CEUR-WS:

  https://ceur-ws.org/Vol-XXXX/ceurart-bundle.zip
  (or any recent CEUR volume bundle)

then place `ceurart.cls` and any required `.sty` files alongside
`working_notes.tex` and compile:

```bash
pdflatex working_notes.tex
pdflatex working_notes.tex   # second pass for cross-refs
```

(If you prefer Overleaf, create a new project, upload
`working_notes.tex`, and add `ceurart` from the CEUR template
gallery.)

## Submitting to EasyChair

Per the CEUR-WS instructions in the call:
- All teams with at least one graded submission must submit working
  notes — even if the score was modest.
- Teams that participated in both subtasks submit one combined report
  (this paper covers both Subtask 1 and Subtask 2).
- Submit the compiled PDF through the ImageCLEFmedical 2026 EasyChair
  page (the call links it).

## Things to update before final submission

- **ORCIDs**: fill in `orcid={}` for both authors in the `\author`
  blocks.
- **Final public-leaderboard scores** for Subtask 1 (BLEU / ROUGE-1/2/L
  / METEOR on the test split). The paper currently reports the
  validation-set results from the deployed system; replace with the
  final test-set numbers once the leaderboard is released.
- **Repo URL**: the paper links the gated HuggingFace submission repo.
  Confirm the URL once the organisers' validation pings have
  completed.
- **Figure 1 (pipeline diagram)**: a one-figure schematic of the
  Subtask-2 pipeline (A → B → C → D) is referenced in the text but
  omitted. Either add a diagram or remove the figure reference.

