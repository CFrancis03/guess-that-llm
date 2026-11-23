"""Level 1 for Name That LLM: MLP on TF-IDF features and a BiLSTM.

Usage:
    python level1_name_that_llm.py --model mlp    --seeds 0 1 2
    python level1_name_that_llm.py --model bilstm --seeds 0 1 2
    python level1_name_that_llm.py --model mlp --seeds 0 1 2 --eval-test   # final run only

Rules (see AGENTS.md): trains on train only; the epoch is chosen on validation
only; the test set is touched only with --eval-test, once per final model.
Every run writes results/level1_<model>_seed<k>_<split>.json, and a summary
results/level1_<model>_summary_<split>.json (mean/std over seeds).
Each run is far below the 30-minute T4 limit (a few minutes at most).
"""
import argparse
import json
import os
import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)

from nll_data import lab_of, load_splits

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["mlp", "bilstm"], required=True)
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--batch-size", type=int, default=64)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--weight-decay", type=float, default=1e-2)
p.add_argument("--hidden", type=int, default=256)
p.add_argument("--dropout", type=float, default=0.5)
p.add_argument("--max-features", type=int, default=50000)  # mlp
p.add_argument("--emb-dim", type=int, default=100)         # bilstm
p.add_argument("--max-len", type=int, default=300)         # bilstm, tokens
p.add_argument("--eval-test", action="store_true",
               help="also evaluate on test (final models only, once)")
p.add_argument("--tag", default="", help="suffix for results files, e.g. a tuning trial name")
args = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
splits = load_splits()
train, val, test = splits["train"], splits["val"], splits["test"]
labels = sorted(train["model"].unique())
l2i = {l: i for i, l in enumerate(labels)}
y_tr, y_va, y_te = (np.array([l2i[m] for m in d["model"]]) for d in (train, val, test))

# ---------------- features (fit on train only) ----------------
if args.model == "mlp":
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True,
                          min_df=2, max_features=args.max_features, dtype=np.float32)
    X_tr = vec.fit_transform(train["text"])
    X_va, X_te = vec.transform(val["text"]), vec.transform(test["text"])
    in_dim = X_tr.shape[1]

    def batch(X, idx):
        return torch.from_numpy(X[idx].toarray()).to(dev)
else:
    tok = lambda s: re.findall(r"\w+|[^\w\s]", s.lower())  # punctuation kept as tokens
    counts = Counter(t for s in train["text"] for t in tok(s))
    vocab = {t: i + 2 for i, (t, c) in enumerate(counts.most_common()) if c >= 2}  # 0 pad, 1 unk

    def encode(texts):
        X = np.zeros((len(texts), args.max_len), dtype=np.int64)
        for i, s in enumerate(texts):
            ids = [vocab.get(t, 1) for t in tok(s)][:args.max_len]
            X[i, :len(ids)] = ids
        return X

    X_tr, X_va, X_te = encode(train["text"]), encode(val["text"]), encode(test["text"])

    def batch(X, idx):
        return torch.from_numpy(X[idx]).to(dev)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_dim, args.hidden), nn.ReLU(),
                                 nn.Dropout(args.dropout), nn.Linear(args.hidden, len(labels)))

    def forward(self, x):
        return self.net(x)


class BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(vocab) + 2, args.emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(args.emb_dim, args.hidden // 2, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(args.dropout)
        self.out = nn.Linear(args.hidden, len(labels))

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)
        h, _ = self.lstm(self.drop(self.emb(x)))
        h = h.masked_fill(~mask, -1e9).max(dim=1).values  # max-pool over real tokens
        return self.out(self.drop(h))


@torch.no_grad()
def predict(net, X):
    net.eval()
    out = []
    for i in range(0, X.shape[0], 256):
        out.append(net(batch(X, np.arange(i, min(i + 256, X.shape[0])))).argmax(1).cpu().numpy())
    return np.concatenate(out)


def metrics(y, pred):
    pred_lab = [lab_of(labels[i]) for i in pred]
    true_lab = [lab_of(labels[i]) for i in y]
    return {
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "lab_accuracy": accuracy_score(true_lab, pred_lab),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y, pred, labels=range(len(labels))).tolist(),
    }


def run(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = (MLP() if args.model == "mlp" else BiLSTM()).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lossf = nn.CrossEntropyLoss()
    best, best_state, best_epoch = -1, None, 0
    rng = np.random.default_rng(seed)
    for ep in range(1, args.epochs + 1):
        net.train()
        order = rng.permutation(X_tr.shape[0])
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            opt.zero_grad()
            lossf(net(batch(X_tr, idx)), torch.from_numpy(y_tr[idx]).to(dev)).backward()
            opt.step()
        bal = balanced_accuracy_score(y_va, predict(net, X_va))  # epoch chosen on validation only
        if bal > best:
            best, best_epoch = bal, ep
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        print(f"seed {seed} epoch {ep:2d} val balanced acc {bal:.3f}")
    net.load_state_dict(best_state)
    return net, best_epoch


os.makedirs("results", exist_ok=True)
name = f"level1_{args.model}{('_' + args.tag) if args.tag else ''}"
evals = [("val", X_va, y_va)] + ([("test", X_te, y_te)] if args.eval_test else [])
per_split = {s: [] for s, _, _ in evals}
for seed in args.seeds:
    net, best_epoch = run(seed)
    for split, X, y in evals:
        m = metrics(y, predict(net, X))
        m.update({"task": "name_that_llm", "level": 1, "model": args.model, "split": split,
                  "seed": seed, "best_epoch": best_epoch, "n": len(y), "config": vars(args)})
        with open(f"results/{name}_seed{seed}_{split}.json", "w") as f:
            json.dump(m, f, indent=2)
        per_split[split].append(m)
        print(f"{split} seed {seed}: balanced accuracy {m['balanced_accuracy']:.3f}, "
              f"accuracy {m['accuracy']:.3f}, macro-F1 {m['macro_f1']:.3f}, "
              f"lab accuracy {m['lab_accuracy']:.3f}")

for split, ms in per_split.items():
    summ = {"task": "name_that_llm", "level": 1, "model": args.model, "split": split,
            "seeds": args.seeds, "config": vars(args)}
    for k in ["balanced_accuracy", "accuracy", "macro_f1", "lab_accuracy"]:
        v = [m[k] for m in ms]
        summ[k + "_mean"], summ[k + "_std"] = float(np.mean(v)), float(np.std(v))
    with open(f"results/{name}_summary_{split}.json", "w") as f:
        json.dump(summ, f, indent=2)
    print(f"{split} mean over seeds: balanced accuracy "
          f"{summ['balanced_accuracy_mean']:.3f} ± {summ['balanced_accuracy_std']:.3f}")
