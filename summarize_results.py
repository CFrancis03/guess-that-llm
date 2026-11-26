"""Read the results files back and print what goes into the README.

    python summarize_results.py level1_mlp level1_bilstm

For each prefix it reads results/<prefix>_seed<k>_{val,test}.json and prints
mean ± sd over seeds, then a look past the average: recall per model and the
share of errors that stay inside the right lab. Level 0 is printed for reference.
Check these numbers against the files by hand once.
"""
import glob
import json
import re
import sys

import numpy as np

from nll_data import lab_of

KEYS = ["balanced_accuracy", "accuracy", "macro_f1", "lab_accuracy"]


def load(prefix, split):
    files = sorted(glob.glob(f"results/{prefix}_seed*_{split}.json"))
    files = [f for f in files if re.search(rf"{re.escape(prefix)}_seed\d+_{split}\.json$", f)]
    return [json.load(open(f)) for f in files], files


def look_past_average(ms):
    """Recall per model and within-lab share of errors, from the summed confusion matrix."""
    labels = ms[0]["labels"]
    C = np.sum([np.array(m["confusion_matrix"]) for m in ms], axis=0)
    recall = np.diag(C) / C.sum(1)
    labs = [lab_of(l) for l in labels]
    wrong = C.sum() - np.trace(C)
    within = sum(C[i, j] for i in range(len(labels)) for j in range(len(labels))
                 if i != j and labs[i] == labs[j])
    pairs = sorted(((C[i, j], labels[i], labels[j]) for i in range(len(labels))
                    for j in range(len(labels)) if i != j), reverse=True)[:5]
    return dict(zip(labels, recall)), within / wrong, pairs


for split in ["val", "test"]:
    m0 = json.load(open(f"results/level0_name_that_llm_{split}.json"))
    print(f"[{split}] level0: " + ", ".join(f"{k} {m0[k]:.3f}" for k in KEYS))

for prefix in sys.argv[1:]:
    print(f"\n=== {prefix} ===")
    for split in ["val", "test"]:
        ms, files = load(prefix, split)
        if not ms:
            print(f"[{split}] no files")
            continue
        seeds = [m["seed"] for m in ms]
        print(f"[{split}] seeds {seeds}, best epochs {[m['best_epoch'] for m in ms]}, "
              f"parameters {ms[0].get('n_params')}")
        for k in KEYS:
            v = np.array([m[k] for m in ms])
            print(f"    {k:18s} {v.mean():.3f} ± {v.std():.3f}   per seed {np.round(v, 3).tolist()}")
        recall, within, pairs = look_past_average(ms)
        print(f"    share of errors inside the right lab: {within:.3f}")
        print("    recall per model (all seeds): " +
              ", ".join(f"{l} {r:.2f}" for l, r in sorted(recall.items(), key=lambda x: x[1])))
        print("    most frequent confusions (true -> predicted, summed over seeds): " +
              "; ".join(f"{a} -> {b} ({int(c)})" for c, a, b in pairs))
