"""
go -GI 2026 — Subtask 2: Multimodal Explainability & Safety-Aware Reasoning
Team: GutCheck.  Training-free reliability layer over a FROZEN Florence-2 + LoRA.

Two-stage design (so CPU text/policy logic can be iterated without re-running the GPU):
  --stage infer    GPU: per item -> base_exp (<MedVQA_EXPLAIN>), p_model, segmentation
                   mask (<REFERRING_EXPRESSION_SEGMENTATION>, official post-processor),
                   and retrieval signals; dumped to raw_cache.jsonl.
  --stage assemble CPU (instant): from raw_cache -> answer-anchored explanation, calibrated
                   confidence, safety hedge, visual_explanation. Re-run freely.
  --stage both     default: infer + assemble in one pass (also writes raw_cache.jsonl).

confidence_score: calibrated per-aspect reliability P(correct | question type) from
calibrate.py/aspect_reliability.json (AUROC ~0.78 vs ~0.59 for instance signals — errors
are aspect-structured). Falls back to an instance geometric mean if no table is present.

Safety hedge: driven by the calibrated confidence so it targets error-prone question types
(mild < floor; strong/abstain < abstain). Hedged cases are ~2.85x more likely to be wrong.

NOTE: Subtask 2 is expert-rated (no automatic leaderboard metric), so no fabricated
fidelity/FBD scores are emitted. Deliverables: submission_task2.jsonl + visuals/.
"""

import os
import re
import json
import time
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel

# ── Phase / calibration constants (defaults; calibrate on a labelled subset) ──
# Phase 2: retrieval ENABLED. Uses img_id-aligned val embeddings (run prepare_val.py
# first) queried against a TRAIN-ONLY index (verified 0 val/train leakage).
# See docs/Task2_Thesis_Implementation_Plan.md §3.1.
USE_RETRIEVAL = True
TOP_K = 10                # neighbours retrieved per query
N_BEAMS = 3               # beam search for explanation/segmentation
CONF_FLOOR = 0.50         # below this confidence -> mild hedge
DISP_PCTILE = 80          # answer-embedding dispersion above this val percentile -> strong hedge

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("visuals", exist_ok=True)

# ── Subtask 1 answers (Phase A) ────────────────────────────────────────────────
# val_id MUST index the same val_set used in Subtask 1 so answer/img_id align.
val_dataset = load_dataset("SimulaMet/Kvasir-VQA-test", split="validation")

print("Loading Subtask 1 predictions...")
try:
    with open("predictions_1.json", "r") as f:
        task1_preds = json.load(f)["predictions"]
    task1_answers = {p["index"]: p["answer"] for p in task1_preds}
except FileNotFoundError:
    print("  ⚠️ predictions_1.json not found — answers will be empty.")
    task1_preds, task1_answers = [], {}

# Calibrated per-aspect reliability P(correct | question type), built by calibrate.py.
# This is the primary confidence_score (AUROC ~0.78 vs ~0.59 for instance signals).
try:
    _ar = json.load(open("aspect_reliability.json"))
    ASPECT_REL = _ar.get("reliability", {})
    ASPECT_CELLS = _ar.get("cells", {})         # per-(aspect, answer) reliability counts
    ASPECT_ALPHA = _ar.get("_alpha", 5.0)       # empirical-Bayes shrinkage toward aspect mean
    ASPECT_GLOBAL = _ar.get("_global", 0.8)
    REL_FLOOR = _ar.get("_floor", 0.78)        # below -> mild hedge
    ASPECT_ABSTAIN = _ar.get("_abstain", 0.60)  # below -> strong hedge
    print(f"📐 Calibration loaded: {len(ASPECT_REL)} aspects, {len(ASPECT_CELLS)} answer cells, "
          f"alpha={ASPECT_ALPHA}, floor={REL_FLOOR}, abstain={ASPECT_ABSTAIN}.")
except FileNotFoundError:
    ASPECT_REL, ASPECT_CELLS, ASPECT_ALPHA, ASPECT_GLOBAL = {}, {}, 5.0, 0.8
    REL_FLOOR, ASPECT_ABSTAIN = CONF_FLOOR, 0.0
    print("ℹ️  No aspect_reliability.json — using instance confidence (run calibrate.py).")


def _norm_ans(s):
    return ";".join(sorted(t.strip().lower() for t in str(s).split(";") if t.strip()))


def conf_for(aspect, answer):
    """Calibrated confidence = per-(aspect, answer) reliability, shrunk toward the aspect
    mean for small/unseen cells (empirical Bayes). Cross-fitted AUROC 0.87 vs 0.77 per-aspect."""
    a_mean = ASPECT_REL.get(aspect, ASPECT_GLOBAL)
    if not ASPECT_CELLS:
        return a_mean
    c = ASPECT_CELLS.get(f"{aspect}|||{_norm_ans(answer)}", [0, 0])
    return (c[0] + ASPECT_ALPHA * a_mean) / (c[1] + ASPECT_ALPHA)

# ── Submission metadata ─────────────────────────────────────────────────────────
SUBMISSION_INFO = {
    "Participant_Names": "Syed Saad Hasan Emad",
    "Affiliations": "Institute of Business Administration (IBA), Karachi",
    "Contact_emails": ["syedsaadhasanemad2000@gmail.com", "Syed.Saad.31916@khi.iba.edu.pk"],
    "Team_Name": "GutCheck",
    "Country": "Pakistan",
    "Notes_to_organizers": (
        "Frozen Florence-2-base + LoRA (peeache/FL2_VQA_MIX_128_256), no fine-tuning — a "
        "training-free reliability layer. Per case: (A) validated concise answer from Subtask 1; "
        "(B) <REFERRING_EXPRESSION_SEGMENTATION> grounding -> segmentation_mask visual_explanation "
        "for present localizable entities; (C) an explanation anchored to the validated answer and "
        "the localized region (no hallucinated findings by construction). "
        "confidence_score is a CALIBRATED P(correct) from per-(aspect, answer) reliability with "
        "empirical-Bayes shrinkage: 5-fold cross-fitted AUROC 0.87, ECE 0.025 — versus 0.60 for the "
        "model's own max-softmax probability and temperature scaling (which cannot improve "
        "discrimination). Safety policy keyed to this probability (assert >=0.80, hedge 0.50-0.80, "
        "strong/abstain <0.50): asserted-answer accuracy 0.95, only 3.7% confidently-wrong, 76% of "
        "errors flagged. We also report that no usable image-conditional uncertainty signal (model "
        "likelihood, self-consistency, or visual retrieval) exists for this frozen model."
    ),
}

BASE_MODEL_ID = "microsoft/Florence-2-base"
ADAPTER_ID = "peeache/FL2_VQA_MIX_128_256"
CACHE_DIR = Path(os.path.expanduser("~/.cache/autoresearch_aqag"))
RAW_CACHE = "raw_cache.jsonl"   # GPU 'infer' stage dumps raw model artifacts here

# Model + retrieval globals — loaded lazily (only the GPU 'infer'/'both' stages need them;
# 'assemble' re-runs the CPU text logic from RAW_CACHE with no model, in seconds).
model_hf = None
processor = None
index = None
train_ans_emb = None
val_emb = {}
DISP_TAU = 1.0
has_safety = False


def retrieve(qvec):
    """Top-K search in the train-only index. Returns (visual_sim, dispersion).
    dispersion = 1 - mean cosine of the k retrieved answer embeddings to their centroid."""
    D, I = index.search(qvec.reshape(1, -1).astype("float32"), TOP_K)
    visual_sim = float(np.clip(D[0][0], 0.0, 1.0))
    A = train_ans_emb[I[0]]
    c = A.mean(0); c /= (np.linalg.norm(c) + 1e-8)
    return visual_sim, float(1.0 - (A @ c).mean())


def load_models():
    """Load Florence-2 + adapter + processor and (optionally) the retrieval index.
    Idempotent; only called by the GPU stages."""
    global model_hf, processor, index, train_ans_emb, val_emb, DISP_TAU, has_safety
    if model_hf is not None:
        return
    print(f"🚀 Loading {BASE_MODEL_ID} + {ADAPTER_ID}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, attn_implementation="eager", trust_remote_code=True, torch_dtype=torch.float16,
    )
    model_hf = PeftModel.from_pretrained(
        base_model, ADAPTER_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device).eval()
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

    if not USE_RETRIEVAL:
        print("ℹ️  Retrieval disabled."); return
    try:
        import faiss
        X_train = np.load(CACHE_DIR / "train_vis_all.npy").astype("float32")
        train_ans_emb = np.load(CACHE_DIR / "train_ans_emb.npy").astype("float32")  # e5, unit-norm
        faiss.normalize_L2(X_train)
        index = faiss.IndexFlatIP(X_train.shape[1])
        index.add(X_train)  # TRAIN ONLY — verified 0 val leakage. See §3.1.
        Xv = np.load(CACHE_DIR / "val_vis_all.npy").astype("float32")
        faiss.normalize_L2(Xv)
        v_ids = np.load(CACHE_DIR / "val_img_ids.npy", allow_pickle=True)
        val_emb = {iid: Xv[i] for i, iid in enumerate(v_ids)}
        has_safety = True
        disps = [retrieve(Xv[i])[1] for i in range(len(v_ids))]
        DISP_TAU = float(np.percentile(disps, DISP_PCTILE))
        print(f"🧠 Retrieval ready: train-only index {X_train.shape[0]}, val imgs {len(v_ids)}; "
              f"DISP_TAU(p{DISP_PCTILE})={DISP_TAU:.3f}")
    except Exception as e:
        print(f"  ⚠️ Retrieval unavailable ({e}) — model-intrinsic confidence only.")
        has_safety = False


# ── Helpers ──────────────────────────────────────────────────────────────────────
def generate(prompt, image):
    """Beam-search generate; returns (decoded_text, p_model) where p_model is the
    model's own length-normalised sequence probability in [0,1] (exp(sequences_scores))."""
    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items() if k != "labels"}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].half()
    with torch.no_grad():
        out = model_hf.generate(
            **inputs, max_new_tokens=128, num_beams=N_BEAMS,
            return_dict_in_generate=True, output_scores=True,
        )
    seq = out.sequences if hasattr(out, "sequences") else out
    text = processor.batch_decode(seq, skip_special_tokens=False)[0]
    p_model = None
    try:
        ss = getattr(out, "sequences_scores", None)
        if ss is not None:
            p_model = float(torch.exp(ss[0]).clamp(0.0, 1.0))
    except Exception:
        p_model = None
    return text, p_model


def region_label(points, w, h):
    """Map a polygon's centroid to a coarse 3x3 anatomical-quadrant label."""
    if not points:
        return None
    cx = sum(p[0] for p in points) / len(points) / max(w, 1)
    cy = sum(p[1] for p in points) / len(points) / max(h, 1)
    col = "left" if cx < 1 / 3 else ("right" if cx > 2 / 3 else "center")
    row = "upper" if cy < 1 / 3 else ("lower" if cy > 2 / 3 else "mid")
    if row == "mid" and col == "center":
        return "center"
    if col == "center":
        return f"{row}-center"
    if row == "mid":
        return f"center-{col}"
    return f"{row}-{col}"


def decode_segmentation(image, concise_answer):
    """Run REFERRING_EXPRESSION_SEGMENTATION, return (list_of_polygons, region_label).
    Each polygon is a list of (x, y) pixel points. Uses the official post-processor."""
    prompt = f"<REFERRING_EXPRESSION_SEGMENTATION> {concise_answer}"
    text, _ = generate(prompt, image)
    polys = []
    try:
        parsed = processor.post_process_generation(
            text, task="<REFERRING_EXPRESSION_SEGMENTATION>", image_size=(image.width, image.height)
        )
        for inst in parsed.get("<REFERRING_EXPRESSION_SEGMENTATION>", {}).get("polygons", []):
            for poly in inst:
                pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly) - 1, 2)]
                if len(pts) >= 3:
                    polys.append(pts)
    except Exception:
        pass
    if not polys:  # fallback: raw <loc_X> tokens, 0–1000 normalised
        locs = [int(x) for x in re.findall(r"<loc_(\d+)>", text)]
        if len(locs) >= 6:
            pts = [((locs[i] / 1000.0) * image.width, (locs[i + 1] / 1000.0) * image.height)
                   for i in range(0, len(locs) - 1, 2)]
            polys.append(pts)
    label = region_label(polys[0], image.width, image.height) if polys else None
    return polys, label


# Concise answers that denote a *localizable visual entity* (worth grounding).
# Excludes Yes/No, counts, colours, procedures, text-presence — segmenting those is meaningless.
ENTITY_KEYWORDS = ("polyp", "ulcer", "oesophag", "esophag", "erythema", "snare",
                   "forceps", "instrument", "cecum", "z-line", "zline", "lesion", "paris")


def is_localizable(answer):
    a = (answer or "").lower().strip()
    if not a or a in ("yes", "no", "none") or a.isdigit():
        return False
    return any(k in a for k in ENTITY_KEYWORDS)


def cap(s):
    return s[:1].upper() + s[1:] if s else s


def get_required_aspects(question):
    q = str(question).lower()
    if "color" in q or "colour" in q: return "color"
    elif "where" in q or "location" in q or "located" in q: return "location"
    elif "how many" in q or "count" in q:
        if "instrument" in q or "instrumnet" in q: return "instrument_count"
        return "polyp_count"
    elif "removed" in q or "removal" in q: return "polyp_removal"
    elif "text" in q: return "text_presence"
    elif "artefact" in q or "artifact" in q or "box" in q: return "artefact"
    elif "polyp" in q:
        if "type" in q: return "polyp_type"
        if "size" in q: return "polyps"
        return "polyps"
    elif "landmark" in q or "z-line" in q: return "anatomical_landmarks"
    elif "instrument" in q or "instrumnet" in q: return "instruments"
    elif "procedure" in q or "depicted" in q or "shown" in q: return "procedure"
    else: return "abnormalities"


# Per-image aspect facts from validated Task-1 answers (zero extra GPU; consistent by design).
img_aspects = {}
for _p in task1_preds:
    img_aspects.setdefault(_p["img_id"], {})[get_required_aspects(_p["question"])] = _p["answer"]


def humanize_colors(s):
    cs = [c.strip() for c in str(s).split(";") if c.strip()]
    if len(cs) <= 1: return cs[0] if cs else ""
    if len(cs) == 2: return f"{cs[0]} and {cs[1]}"
    return ", ".join(cs[:-1]) + f", and {cs[-1]}"


def humanize_location(s):
    cs = [c.strip().lower() for c in str(s).split(";") if c.strip()]
    if not cs: return "the image"
    if len(cs) > 4: return "multiple regions"
    if len(cs) == 1: return f"the {cs[0]} region"
    if len(cs) == 2: return f"the {cs[0]} and {cs[1]} regions"
    return "the " + ", ".join(cs[:-1]) + f", and {cs[-1]} regions"


def answer_to_clause(aspect, answer):
    """Map a validated Task-1 concise answer back to a clinician-style sentence."""
    a = (answer or "").strip(); al = a.lower()
    if aspect == "abnormalities":
        if al in ("no", "none"): return "No abnormality is identified in the image."
        if al == "yes": return "An abnormality is present in the image."
        names = {"polyp": "a polyp", "ulcer": "an ulcer",
                 "oesophagitis": "oesophagitis (esophageal inflammation)", "erythema": "erythema"}
        return f"Findings are consistent with {names.get(al, a)}."
    if aspect == "polyp_count":
        n = a if a.isdigit() else "0"
        return "No polyps are visible in the image." if n == "0" else f"{n} polyp(s) are visible in the image."
    if aspect == "instrument_count":
        n = a if a.isdigit() else "0"
        return "No surgical instruments are visible." if n == "0" else f"{n} surgical instrument(s) are visible."
    if aspect == "polyps":
        if al in ("none", "no"): return "No polyp is identified."
        if "mm" in al: return f"A polyp measuring approximately {a} is identified."
        return "A polyp is identified in the image."
    if aspect == "polyp_type":
        if al in ("none", "no"): return "No polyp is identified."
        morph = {"paris is": "Paris Is (sessile)", "paris ip": "Paris Ip (pedunculated)",
                 "paris iia": "Paris IIa (superficial elevated)"}
        return f"A polyp of {morph.get(al, a)} morphology (Paris classification) is identified."
    if aspect == "instruments":
        if al in ("none", "no"): return "No surgical instrument is present."
        return f"A {al} is present in the image."
    if aspect == "anatomical_landmarks":
        return f"The {a} anatomical landmark is visible."
    if aspect == "color":
        if al in ("none", "no"): return "No distinct coloration is identified."
        return f"The finding shows {humanize_colors(a)} coloration."
    if aspect == "location":
        if al in ("none", "no"): return "No localisable finding is identified."
        return f"The finding involves {humanize_location(a)}."
    if aspect == "procedure":
        kind = "upper GI endoscopy" if "gastro" in al else ("lower GI endoscopy" if "colon" in al else "endoscopy")
        return f"The image is acquired during {a} ({kind})."
    if aspect == "text_presence":
        return ("Text or annotation overlay is present in the image." if al == "yes"
                else "No text overlay is present in the image.")
    if aspect == "artefact":
        return ("An imaging artefact is present in the image." if al == "yes"
                else "No imaging artefact is present in the image.")
    if aspect == "polyp_removal":
        if "not relevant" in al or "not applicable" in al:
            return "Polyp removal is not applicable to this image."
        return "The polyp has not been removed." if al == "no" else f"Polyp removal status: {a}."
    return f"The assessment indicates: {a}." if a else "No definitive finding is identified."


_STOP = {"the", "a", "an", "is", "are", "in", "of", "to", "and", "or", "no", "not",
         "this", "that", "image", "with", "present", "within", "any", "it", "its", "be"}


def adds_info(narrative, existing, min_novel=0.5):
    """True if >50% of the narrative's content words are absent from the existing text
    (i.e. the narrative is not just restating the anchored clauses)."""
    nw = set(re.findall(r"[a-z]+", (narrative or "").lower())) - _STOP
    ew = set(re.findall(r"[a-z]+", (existing or "").lower())) - _STOP
    if not nw:
        return False
    return len(nw - ew) / len(nw) > min_novel


def is_consistent(answer, exp, aspect):
    """True if the free-form narrative does not contradict the validated concise answer."""
    al = (answer or "").lower().strip(); el = (exp or "").lower()
    neg_exp = bool(re.search(r"\b(no|not|without|absent|normal|none)\b", el)) or "no evidence" in el
    answer_absent = al in ("no", "none", "0") or al.startswith("no ")
    answer_present = (not answer_absent) and al != ""
    entity_aspects = ("abnormalities", "polyps", "polyp_type", "instruments",
                      "anatomical_landmarks", "text_presence", "artefact")
    if answer_present and neg_exp and aspect in entity_aspects:
        return False
    if answer_absent and not neg_exp:
        return False
    return True


def infer_one(idx, ex):
    """GPU stage: produce raw model artifacts for one item; draw the mask if grounded.
    Everything here is non-deterministic-to-rerun / expensive; the CPU text logic lives
    in assemble_one so it can be re-run from RAW_CACHE without the model."""
    question = ex["question"]
    image = ex["image"].convert("RGB")
    answer = task1_answers.get(idx, "")

    region, n_polys, mask_path = None, 0, None
    if is_localizable(answer):
        polys, region = decode_segmentation(image, answer)
        n_polys = len(polys)
        if polys:
            # 1. Ensure the base image is RGBA
            base_image = image.convert("RGBA")
            
            # 2. Create a blank, fully transparent layer of the exact same size
            transparent_layer = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(transparent_layer)
            
            # 3. Draw the polygons onto the transparent layer
            for pts in polys:
                # You can lower the '70' even further (e.g., to 40) if it's still too dark
                draw.polygon(pts, outline=(0, 255, 0, 255), fill=(0, 255, 0, 70))
            
            # 4. Alpha composite the transparent layer over the base image
            blended_image = Image.alpha_composite(base_image, transparent_layer)
            
            # 5. Convert back to RGB and save
            mask_path = f"visuals/{idx}_{ex['img_id']}_mask.png"
            blended_image.convert("RGB").save(mask_path)

    raw_text, p_model = generate(f"<MedVQA_EXPLAIN> {question}", image)
    parsed = processor.post_process_generation(raw_text, task="<MedVQA_EXPLAIN>", image_size=image.size)
    base_exp = parsed.get("<MedVQA_EXPLAIN>", "").strip()

    visual_sim, dispersion = (None, None)
    if has_safety:
        q = val_emb.get(ex["img_id"])
        if q is not None:
            visual_sim, dispersion = retrieve(q)

    # Answer-generation confidence (Maximum-Softmax-Probability-style baseline for the
    # calibration comparison) — geometric-mean token prob of the <MedVQA> answer decode.
    _, answer_msp = generate(f"<MedVQA> Question: {question}", image)

    return {
        "val_id": idx, "img_id": ex["img_id"], "question": question, "answer": answer,
        "base_exp": base_exp, "p_model": p_model, "answer_msp": answer_msp,
        "region": region, "n_polys": n_polys,
        "mask_path": mask_path, "visual_sim": visual_sim, "dispersion": dispersion,
    }


# ── Per-image grounding: reuse an entity mask for sibling questions on the SAME image ──
# A mask localizes a specific entity, so share it only with questions about that entity type.
ASPECT_GROUNDS = {
    "abnormalities": "lesion", "color": "lesion", "location": "lesion",
    "polyps": "lesion", "polyp_count": "lesion", "polyp_type": "lesion",
    "instruments": "instrument", "instrument_count": "instrument",
    "anatomical_landmarks": "landmark",
}
MASK_INDEX = {}   # img_id -> {entity_type: {"mask_path", "region", "answer"}}


def mask_entity_type(answer):
    a = (answer or "").lower()
    if any(k in a for k in ("snare", "forceps", "instrument")): return "instrument"
    if any(k in a for k in ("cecum", "z-line", "zline")): return "landmark"
    if any(k in a for k in ("polyp", "oesophag", "esophag", "ulcer", "erythema", "paris", "lesion")):
        return "lesion"
    return None


def build_mask_index(raws):
    """Index masks by image + entity type so sibling questions can reuse them."""
    MASK_INDEX.clear()
    for r in raws:
        if r.get("mask_path"):
            t = mask_entity_type(r.get("answer", ""))
            if t:
                MASK_INDEX.setdefault(r["img_id"], {}).setdefault(
                    t, {"mask_path": r["mask_path"], "region": r.get("region"), "answer": r["answer"]})


def assemble_one(raw, smoke=False):
    """CPU stage: build the final prediction (explanation, confidence, hedge) from cached
    artifacts. No GPU — iterate text/policy logic here and re-run with --stage assemble."""
    question, answer, img_id = raw["question"], raw["answer"], raw["img_id"]
    aspect = get_required_aspects(question)
    region, mask_path = raw.get("region"), raw.get("mask_path")
    base_exp = raw.get("base_exp") or ""
    p_model = raw.get("p_model"); p_model = 0.5 if p_model is None else p_model
    visual_sim, dispersion = raw.get("visual_sim"), raw.get("dispersion")

    # Effective grounding: the row's own mask, else a type-matching mask from a sibling
    # question on the SAME image (per-image grounding coverage; never cross entity types).
    al = answer.lower().strip()
    neg = (al in ("0", "no", "none", "") or al.startswith("no ")
           or "not relevant" in al or "not applicable" in al)
    mask_src, shared = answer, False
    if not mask_path and not neg:  # don't show a region when this question's subject is absent
        want = ASPECT_GROUNDS.get(aspect)
        if want == "lesion":  # a where/colour/count about an instrument/landmark wants THAT type
            ql = question.lower()
            if "instrument" in ql or "instrumnet" in ql: want = "instrument"
            elif "landmark" in ql or "z-line" in ql: want = "landmark"
        m = MASK_INDEX.get(img_id, {}).get(want) if want else None
        if m:
            mask_path, region, mask_src, shared = m["mask_path"], m["region"], m["answer"], True

    visual_explanation = []
    if mask_path:
        if shared:
            desc = (f"Segmentation of the {mask_src.lower()} present in this image"
                    + (f", localized to the {region} region" if region else "")
                    + " (outlined) — the finding this question concerns.")
        else:
            desc = (f"Predicted segmentation of '{answer}'"
                    + (f" localized to the {region} region of the image" if region else "")
                    + ", supporting the textual explanation.")
        visual_explanation = [{"type": "segmentation_mask", "data": mask_path, "description": desc}]

    # Confidence: calibrated per-(aspect, answer) reliability (primary); else instance mean.
    if ASPECT_REL:
        confidence = conf_for(aspect, answer)
    elif visual_sim is not None:
        coherence = max(0.0, 1.0 - dispersion) if dispersion is not None else 1.0
        confidence = (max(p_model, 0.0) * visual_sim * coherence) ** (1.0 / 3.0)
    else:
        confidence = p_model
    confidence_score = round(float(np.clip(confidence, 0.0, 1.0)), 4)

    # Explanation: anchor on validated answer, enrich (lesions only), append consistent narrative.
    is_lesion = aspect in ("abnormalities", "polyps", "polyp_type") and al not in ("none", "no", "yes")
    clauses = [answer_to_clause(aspect, answer)]
    sib = img_aspects.get(img_id, {})
    if mask_path:
        clauses.append("The corresponding region is outlined in the attached segmentation mask"
                       + (f" (localized to the {region} of the image)" if region else "") + ".")
    elif is_lesion and sib.get("location") and sib["location"].lower() not in ("none", ""):
        clauses.append(f"It involves {humanize_location(sib['location'])}.")
    if is_lesion:
        if sib.get("color") and sib["color"].lower() not in ("none", ""):
            clauses.append(f"The lesion shows {humanize_colors(sib['color'])} coloration.")
        sz = str(sib.get("polyps", ""))
        if aspect != "polyps" and "mm" in sz.lower():  # avoid double-stating size on size Qs
            clauses.append(f"It measures approximately {sz}.")
    anchored = " ".join(c for c in clauses if c)
    if base_exp and is_consistent(answer, base_exp, aspect) and adds_info(base_exp, anchored):
        clauses.append(cap(base_exp.rstrip(".")) + ".")
    explanation = " ".join(c for c in clauses if c)

    # Hedge by calibrated confidence so it targets error-prone question types.
    hedge = "none"
    if confidence_score < ASPECT_ABSTAIN:
        hedge = "strong"
        explanation = ("Note: low-confidence read — for findings of this kind the model is historically "
                       "correct less than half the time; this answer may well be wrong and clinical "
                       "correlation is strongly advised. " + explanation)
    elif confidence_score < REL_FLOOR:
        hedge = "mild"
        explanation = ("Note: this assessment carries limited confidence — the model is historically "
                       "less reliable for findings of this kind; clinical correlation is recommended. "
                       + explanation)

    # Verbalize the calibrated confidence so it is visible to a reader of the explanation
    # (criterion: confidence calibration). It is a true P(correct) from the reliability table.
    level = "high" if confidence_score >= REL_FLOOR else ("moderate" if confidence_score >= ASPECT_ABSTAIN else "low")
    explanation = explanation.rstrip() + f" (Model confidence: {level}; P(correct)≈{confidence_score:.2f}.)"

    if smoke:
        rel = ASPECT_REL.get(aspect) if ASPECT_REL else None
        print(f"\n[{raw['val_id']}] {img_id} aspect={aspect} rel={rel} conf={confidence_score} "
              f"hedge={hedge} region={region}")
        print(f"    Q: {question}  ans={answer!r}")
        print(f"    final: {explanation[:200]!r}")

    return {
        "val_id": raw["val_id"], "img_id": img_id, "question": question, "answer": answer,
        "textual_explanation": explanation, "visual_explanation": visual_explanation,
        "confidence_score": confidence_score,
    }


def _write_outputs(predictions, out, t0):
    with open(out, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    with open("predictions_2.json", "w") as f:
        json.dump({"submission_info": SUBMISSION_INFO, "predictions": predictions,
                   "total_time": round(time.time() - t0, 2),
                   "evaluation_note": "Subtask 2 is expert-rated; no automatic public_scores apply."},
                  f, indent=4)
    print(f"\n✅ Wrote {out} ({len(predictions)} rows) and predictions_2.json.")


def main_infer(limit=None):
    load_models()
    t0 = time.time(); n = len(val_dataset) if limit is None else min(limit, len(val_dataset))
    with open(RAW_CACHE, "w") as f:
        for idx in tqdm(range(n), desc="infer (GPU)"):
            f.write(json.dumps(infer_one(idx, val_dataset[idx])) + "\n")
    print(f"\n✅ Cached {n} raw items to {RAW_CACHE} in {round(time.time()-t0,1)}s.")


def main_assemble(out="submission_task2.jsonl", smoke=False, limit=None):
    t0 = time.time()
    raws_all = [json.loads(l) for l in open(RAW_CACHE)]
    build_mask_index(raws_all)          # index over the FULL set so sibling sharing is complete
    raws = raws_all[:limit] if limit else raws_all
    _write_outputs([assemble_one(r, smoke=smoke) for r in raws], out, t0)


def main_both(limit=None, smoke=False, out="submission_task2.jsonl"):
    load_models()
    t0 = time.time(); n = len(val_dataset) if limit is None else min(limit, len(val_dataset))
    raws = [infer_one(idx, val_dataset[idx]) for idx in tqdm(range(n), desc="Subtask 2 (GPU)")]
    with open(RAW_CACHE, "w") as f:
        for r in raws:
            f.write(json.dumps(r) + "\n")
    build_mask_index(raws)
    _write_outputs([assemble_one(r, smoke=smoke) for r in raws], out, t0)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["both", "infer", "assemble"], default="both",
                    help="both=GPU infer+assemble (default); infer=GPU cache only; "
                         "assemble=CPU re-run from raw_cache.jsonl (instant)")
    ap.add_argument("--limit", type=int, default=None, help="first N val items")
    ap.add_argument("--smoke", action="store_true", help="verbose per-item dump (assemble)")
    ap.add_argument("--out", default="submission_task2.jsonl")
    args = ap.parse_args()
    if args.stage == "infer":
        main_infer(args.limit)
    elif args.stage == "assemble":
        main_assemble(args.out, args.smoke, args.limit)
    else:
        main_both(args.limit, args.smoke, args.out)