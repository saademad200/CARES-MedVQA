"""make_figures.py — thesis figures from eval_task2_report.json (run eval_task2.py first)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rep = json.load(open("eval_task2_report.json"))
xf = rep["cross_fitted"]; rel = rep["reliability_xf"]

fig, ax = plt.subplots(figsize=(5.2, 5.2))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
conf = [r["conf"] for r in rel]; acc = [r["acc"] for r in rel]; n = [r["n"] for r in rel]
ax.scatter(conf, acc, s=[max(20, v / 3) for v in n], c="#1f77b4", alpha=0.8, edgecolors="k", zorder=3)
for r in rel:
    ax.annotate(f"n={r['n']}", (r["conf"], r["acc"]), textcoords="offset points",
                xytext=(6, -10), fontsize=7, color="#444")
ax.plot(conf, acc, "-", color="#1f77b4", alpha=0.5, zorder=2)
ax.set_xlabel("Confidence (calibrated per-aspect reliability)")
ax.set_ylabel("Empirical accuracy")
ax.set_title(f"Reliability diagram — cross-fitted (5-fold)\n"
             f"ECE={xf['ECE']}  AUROC={xf['AUROC_conf_vs_correct']}  "
             f"acc={rep['accuracy']}  n={rep['n']}")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig("reliability_diagram.png", dpi=160)
print("✅ Wrote reliability_diagram.png")
