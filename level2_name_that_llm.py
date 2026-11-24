"""Level 2 for Name That LLM: fine-tune a pretrained transformer (DistilRoBERTa).

Usage (Colab, T4 GPU):
    python level2_name_that_llm.py --seeds 0 1 2                  # tune on validation
    python level2_name_that_llm.py --lr 3e-5 --tag lr3e-5         # another trial
    python level2_name_that_llm.py --seeds 0 1 2 --eval-test      # final model, once

Rules (see AGENTS.md): trains on train only; the epoch is chosen on validation
only; the test set is touched only with --eval-test, once for the final model.
Weights are kept in memory and never saved. Every run writes
results/level2_<name>_seed<k>_<split>.json and a summary over seeds.
About 1-2 minutes per epoch on a T4 with the defaults, so 3 seeds x 6 epochs
stays under the 30-minute limit.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from bootstrap import bootstrap_ci
from nll_data import lab_of, load_splits

p = argparse.ArgumentParser()
p.add_argument("--model-name", default="distilroberta-base")
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
p.add_argument("--epochs", type=int, default=6)
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--lr", type=float, default=5e-5)
p.add_argument("--weight-decay", type=float, default=0.01)
p.add_argument("--warmup", type=float, default=0.1, help="fraction of steps")
p.add_argument("--max-len", type=int, default=384, help="tokens; passages are <= ~350")
p.add_argument("--eval-test", action="store_true",
               help="also evaluate on test (final model only, once)")
p.add_argument("--tag", default="", help="suffix for results files, e.g. a tuning trial name")
args = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = dev == "cuda"
splits = load_splits()
train, val, test = splits["train"], splits["val"], splits["test"]
labels = sorted(train["model"].unique())
l2i = {l: i for i, l in enumerate(labels)}
y_tr, y_va, y_te = (np.array([l2i[m] for m in d["model"]]) for d in (train, val, test))

# The tokenizer is cased and keeps punctuation such as em dashes.
tok = AutoTokenizer.from_pretrained(args.model_name)
enc = {name: tok(list(d["text"]), truncation=True, max_length=args.max_len)
       for name, d in [("train", train), ("val", val), ("test", test)]}


def batch(name, idx):
    feats = [{k: enc[name][k][i] for k in ("input_ids", "attention_mask")} for i in idx]
    out = tok.pad(feats, return_tensors="pt")  # pad to the longest in the batch
    return {k: v.to(dev) for k, v in out.items()}


@torch.no_grad()
def predict(net, name):
    net.eval()
    n, out = len(enc[name]["input_ids"]), []
    for i in range(0, n, 64):
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = net(**batch(name, range(i, min(i + 64, n)))).logits
        out.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(out)


def metrics(y, pred, prompt_ids):
    pred_lab = [lab_of(labels[i]) for i in pred]
    true_lab = [lab_of(labels[i]) for i in y]
    return {
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        # 95% interval from resampling prompts
        "balanced_accuracy_ci95": bootstrap_ci(y, pred, prompt_ids),
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "lab_accuracy": accuracy_score(true_lab, pred_lab),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y, pred, labels=range(len(labels))).tolist(),
    }


def run(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(labels)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * ((len(y_tr) + args.batch_size - 1) // args.batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(seed)
    best, best_state, best_epoch = -1, None, 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        net.train()
        order, total = rng.permutation(len(y_tr)), 0.0
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = net(**batch("train", idx),
                           labels=torch.from_numpy(y_tr[idx]).to(dev)).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(idx)
        bal = balanced_accuracy_score(y_va, predict(net, "val"))  # epoch chosen on validation only
        if bal > best:
            best, best_epoch = bal, ep
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        print(f"seed {seed} epoch {ep} train loss {total / len(y_tr):.3f} "
              f"val balanced acc {bal:.3f} ({time.time() - t0:.0f}s)")
    net.load_state_dict(best_state)
    return net, best_epoch


os.makedirs("results", exist_ok=True)
name = "level2_" + args.model_name.split("/")[-1] + (f"_{args.tag}" if args.tag else "")
evals = [("val", y_va, val["prompt_id"])] + \
    ([("test", y_te, test["prompt_id"])] if args.eval_test else [])
per_split = {s: [] for s, *_ in evals}
for seed in args.seeds:
    net, best_epoch = run(seed)
    for split, y, pids in evals:
        m = metrics(y, predict(net, split), pids)
        m.update({"task": "name_that_llm", "level": 2, "model": args.model_name, "split": split,
                  "seed": seed, "best_epoch": best_epoch, "n": len(y), "config": vars(args)})
        with open(f"results/{name}_seed{seed}_{split}.json", "w") as f:
            json.dump(m, f, indent=2)
        per_split[split].append(m)
        ci = m["balanced_accuracy_ci95"]
        print(f"{split} seed {seed}: balanced accuracy {m['balanced_accuracy']:.3f} "
              f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}], accuracy {m['accuracy']:.3f}, "
              f"macro-F1 {m['macro_f1']:.3f}, lab accuracy {m['lab_accuracy']:.3f}")
    del net
    torch.cuda.empty_cache()

for split, ms in per_split.items():
    summ = {"task": "name_that_llm", "level": 2, "model": args.model_name, "split": split,
            "seeds": args.seeds, "config": vars(args)}
    for k in ["balanced_accuracy", "accuracy", "macro_f1", "lab_accuracy"]:
        v = [m[k] for m in ms]
        summ[k + "_mean"], summ[k + "_std"] = float(np.mean(v)), float(np.std(v))
    summ["balanced_accuracy_ci95_mean_of_seeds"] = {
        b: float(np.mean([m["balanced_accuracy_ci95"][b] for m in ms])) for b in ["ci_low", "ci_high"]}
    with open(f"results/{name}_summary_{split}.json", "w") as f:
        json.dump(summ, f, indent=2)
    print(f"{split} mean over seeds: balanced accuracy "
          f"{summ['balanced_accuracy_mean']:.3f} ± {summ['balanced_accuracy_std']:.3f}")
