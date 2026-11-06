"""Load the Name That LLM data and split it by prompt (fixed seed).

All answers to one prompt stay in the same split, so every model is
tested on prompts it has never seen. Do not change SEED or the split sizes.
"""
import numpy as np
import pandas as pd

DATA_PATH = "data/train.csv"
SEED = 42
N_TRAIN, N_VAL = 100, 25  # prompts; the remaining 25 are test

LAB = {
    "claude": "Anthropic", "gemini": "Google", "glm": "ZAI",
    "gpt": "OpenAI", "kimi": "Moonshot AI", "qwen": "Qwen",
}


def lab_of(model):
    for prefix, lab in LAB.items():
        if model.startswith(prefix):
            return lab
    raise ValueError(f"unknown lab for model {model}")


def load_splits(path=DATA_PATH):
    """Return a dict {'train', 'val', 'test'} of DataFrames with columns
    id, prompt_id, text, model, lab."""
    df = pd.read_csv(path)
    df["lab"] = df["model"].map(lab_of)
    prompts = np.array(sorted(df["prompt_id"].unique()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(prompts)
    groups = {
        "train": set(prompts[:N_TRAIN]),
        "val": set(prompts[N_TRAIN:N_TRAIN + N_VAL]),
        "test": set(prompts[N_TRAIN + N_VAL:]),
    }
    return {name: df[df["prompt_id"].isin(p)].reset_index(drop=True)
            for name, p in groups.items()}
