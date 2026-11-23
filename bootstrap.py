"""Bootstrap confidence interval for balanced accuracy, resampling prompts.

All passages for one prompt are kept together, so the interval reflects how
much the score depends on which 25 prompts happen to be in the split.
"""
import numpy as np
from sklearn.metrics import balanced_accuracy_score


def bootstrap_ci(y_true, y_pred, prompt_ids, n_boot=1000, alpha=0.05, seed=0):
    """Return {'ci_low', 'ci_high', 'n_boot', 'unit'} for balanced accuracy."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    prompt_ids = np.asarray(prompt_ids)
    prompts = np.unique(prompt_ids)
    rows = {p: np.flatnonzero(prompt_ids == p) for p in prompts}
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[p] for p in rng.choice(prompts, len(prompts))])
        scores.append(balanced_accuracy_score(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"ci_low": float(lo), "ci_high": float(hi), "n_boot": n_boot, "unit": "prompt"}
