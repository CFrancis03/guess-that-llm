"""Data check for Name That LLM: counts per model and split, and samples.md."""
import pandas as pd
from nll_data import load_splits

splits = load_splits()

# no prompt may appear in two splits
p = {k: set(v["prompt_id"]) for k, v in splits.items()}
assert not (p["train"] & p["val"] or p["train"] & p["test"] or p["val"] & p["test"])

print("Prompts per split:", {k: len(v) for k, v in p.items()})
print("Passages per split:", {k: len(v) for k, v in splits.items()})

counts = pd.DataFrame({k: v["model"].value_counts() for k, v in splits.items()}).fillna(0).astype(int)
counts.insert(0, "lab", [splits["train"].set_index("model")["lab"].groupby(level=0).first()[m] for m in counts.index])
counts = counts.sort_values(["lab"]).rename_axis("model")
print("\nPassages per model and split:")
print(counts.to_string())
missing = (counts[["train", "val", "test"]] == 0).any(axis=1)
print("\nEvery model in every split:", not missing.any())

print("\nPassages per lab and split:")
print(pd.DataFrame({k: v["lab"].value_counts() for k, v in splits.items()}).to_string())

# markdown table for the README
lines = ["| Model | Lab | Train | Validation | Test |", "|---|---|---:|---:|---:|"]
for m, r in counts.iterrows():
    lines.append(f"| {m} | {r['lab']} | {r['train']} | {r['val']} | {r['test']} |")
tot = counts[["train", "val", "test"]].sum()
lines.append(f"| **Total** | | {tot['train']} | {tot['val']} | {tot['test']} |")
print("\nREADME table:\n" + "\n".join(lines))

# three training examples of each model
train = splits["train"]
out = ["# Samples: three training passages per model\n"]
for m in counts.index:
    out.append(f"## {m} ({counts.loc[m, 'lab']})\n")
    for _, r in train[train["model"] == m].sample(3, random_state=0).iterrows():
        out.append(f"**{r['id']}** (prompt {r['prompt_id']}, {len(r['text'].split())} words)\n")
        out.append("> " + r["text"].replace("\n", "\n> ") + "\n")
with open("samples.md", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("\nWrote samples.md")
