"""
narrate.py — PROTOTYPE: gated LLM clinical narrator (Qwen2.5-VL-7B-Instruct, 4-bit, text-only).

Goal: test whether an LLM can make our explanations more *fluent* (criterion 2) WITHOUT
introducing findings (preserving the no-hallucination guarantee). It only REPHRASES our
already-validated template explanation; a faithfulness gate rejects any output that adds a
finding/diagnosis/measurement not present in the input and falls back to the template.

Run as a prototype on a sample to judge prose quality + gate pass-rate BEFORE shipping:
  python narrate.py --n 24
"""
import os, re, json, argparse
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
# Finding/diagnosis vocabulary: if any appears in the LLM output but NOT in the input facts,
# the rewrite has hallucinated -> reject and fall back to the template.
FINDING_TERMS = ["polyp", "ulcer", "oesophag", "esophag", "erythema", "cancer", "malignan",
                 "carcinoma", "bleed", "haemorrh", "hemorrh", "inflammat", "diverticul", "varice",
                 "stricture", "tumor", "tumour", "mass", "nodule", "barrett", "metaplas", "dysplas",
                 "necrosis", "perforation", "obstruction", "colitis", "erosion", "angioectasia"]

HEDGE_PREFIXES = ("Note: low-confidence read", "Note: this assessment carries limited confidence")


def split_explanation(exp):
    """Return (hedge_prefix, core, conf_tag) so we only rephrase the factual core."""
    hedge = ""
    for p in HEDGE_PREFIXES:
        if exp.startswith(p):
            cut = exp.index(". ") + 2
            hedge, exp = exp[:cut], exp[cut:]
            break
    conf = ""
    m = re.search(r"\s*\(Model confidence:.*$", exp)
    if m:
        conf, exp = exp[m.start():].strip(), exp[:m.start()].strip()
    return hedge, exp.strip(), conf


def gate(rewrite, core):
    """Reject rewrites that add findings/measurements absent from the factual core."""
    if not rewrite or len(rewrite.split()) > 60:
        return False, "length"
    rl, cl = rewrite.lower(), core.lower()
    for t in FINDING_TERMS:
        if t in rl and t not in cl:
            return False, f"added '{t}'"
    # any mm-measurement in the rewrite must be present verbatim in the core
    for mm in re.findall(r"\d+\s*-?\s*\d*\s*mm|<\s*\d+\s*mm|>\s*\d+\s*mm", rl):
        if mm.replace(" ", "") not in cl.replace(" ", ""):
            return False, "added measurement"
    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--jsonl", default="submission_task2.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.jsonl)]
    # diverse sample: spread across the leading words of the question
    seen, sample = set(), []
    for r in rows:
        key = (r["question"][:18], bool(r["visual_explanation"]))
        if key in seen:
            continue
        seen.add(key); sample.append(r)
        if len(sample) >= args.n:
            break

    print(f"Loading {MODEL_ID} in 4-bit...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="cuda", torch_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    SYS = ("You are a careful GI-endoscopy reporting assistant. Rewrite the given finding as ONE "
           "fluent, professional clinical sentence. Use ONLY the information provided — do NOT add "
           "any diagnosis, finding, measurement, location, or colour that is not stated. No preamble.")
    passed = 0
    for r in sample:
        hedge, core, conf = split_explanation(r["textual_explanation"])
        prompt = f"Finding: {core}\n\nRewrite:"
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": prompt}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = processor(text=[text], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=90, do_sample=False)
        gen = processor.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        ok, why = gate(gen, core)
        passed += ok
        final = (hedge + (gen if ok else core) + (" " + conf if conf else "")).strip()
        print(f"\n── ans={r['answer']!r}  gate={'PASS' if ok else 'FALLBACK('+why+')'}")
        print(f"  TEMPLATE: {core}")
        print(f"  LLM     : {gen}")
    print(f"\n=== gate pass-rate: {passed}/{len(sample)} = {100*passed/len(sample):.0f}% ===")


if __name__ == "__main__":
    main()
