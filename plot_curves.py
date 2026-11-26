"""Training curves for one seed: figures/<prefix>_seed<k>_curves.png.

    python plot_curves.py level1_mlp 0

Reads the per-epoch history saved in results/<prefix>_seed<k>_val.json.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

prefix = sys.argv[1] if len(sys.argv) > 1 else "level1_mlp"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
m = json.load(open(f"results/{prefix}_seed{seed}_val.json"))
h = m["history"]
ep = [r["epoch"] for r in h]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
a1.plot(ep, [r["train_loss"] for r in h], label="train loss")
a1.plot(ep, [r["val_loss"] for r in h], label="validation loss")
a1.set_xlabel("epoch"); a1.set_ylabel("cross-entropy"); a1.legend()
a2.plot(ep, [r["val_balanced_accuracy"] for r in h], color="C2", label="validation")
lv0 = json.load(open("results/level0_name_that_llm_val.json"))["balanced_accuracy"]
a2.axhline(lv0, color="gray", ls="--", lw=1, label=f"Level 0 ({lv0:.3f})")
a2.axvline(m["best_epoch"], color="C3", ls=":", lw=1, label=f"kept epoch {m['best_epoch']}")
a2.set_xlabel("epoch"); a2.set_ylabel("balanced accuracy"); a2.legend()
for a in (a1, a2):
    a.grid(alpha=0.3)
fig.suptitle(f"{prefix}, seed {seed}")
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
out = f"figures/{prefix}_seed{seed}_curves.png"
fig.savefig(out, dpi=150)
print("saved", out)
