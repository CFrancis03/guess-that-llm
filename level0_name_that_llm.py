"""Level 0 for Name That LLM: TF-IDF on character n-grams + logistic regression.

Trains on train, evaluates on validation and test (test once), and writes
results/level0_name_that_llm_val.json and results/level0_name_that_llm_test.json.
"""
import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)
from sklearn.pipeline import make_pipeline

from nll_data import lab_of, load_splits

splits = load_splits()
train, val, test = splits["train"], splits["val"], splits["test"]
labels = sorted(train["model"].unique())

model = make_pipeline(
    TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2),
    LogisticRegression(C=10, max_iter=3000),
)
model.fit(train["text"], train["model"])


def metrics(df):
    pred = model.predict(df["text"])
    pred_lab = [lab_of(p) for p in pred]
    return {
        "balanced_accuracy": balanced_accuracy_score(df["model"], pred),
        "accuracy": accuracy_score(df["model"], pred),
        "macro_f1": f1_score(df["model"], pred, average="macro"),
        # share of predictions that name a model from the right lab
        "lab_accuracy": accuracy_score(df["lab"], pred_lab),
        "labels": labels,
        "confusion_matrix": confusion_matrix(df["model"], pred, labels=labels).tolist(),
    }


os.makedirs("results", exist_ok=True)
for name, df in [("val", val), ("test", test)]:
    m = metrics(df)
    m.update({"task": "name_that_llm", "level": 0,
              "model": "tfidf_char_2-5_logreg_C10", "split": name, "n": len(df)})
    path = f"results/level0_name_that_llm_{name}.json"
    with open(path, "w") as f:
        json.dump(m, f, indent=2)
    print(f"{name}: balanced accuracy {m['balanced_accuracy']:.3f}, "
          f"accuracy {m['accuracy']:.3f}, macro-F1 {m['macro_f1']:.3f}, "
          f"lab accuracy {m['lab_accuracy']:.3f}  -> {path}")
