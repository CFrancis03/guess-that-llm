# Name That LLM: Which Model Wrote This Passage?

**Author:** Colton Francis · CSCI 490-B9 Deep Learning, NIU, Fall 2026

**AI tool use:** Claude (claude.ai) helped draft this README and write the scripts. I chose the task, split, models and Level 3 question, and ran the scripts myself.

## 1. Task

Input: a 40–250 word passage written by an LLM. Output: which of 18 models wrote it (from OpenAI, Google, Anthropic, Qwen, ZAI and Moonshot AI).

**Example** (prompt `p090`): *Beneath a ferocious sky warped by monstrous gravity, the neutron star’s crust endures unimaginable torment. Here, atomic lattices violently collapse, fusing electrons and protons into an incomprehensibly dense nuclear fluid where a thimbleful outweighs mountains. Spacetime bends savagely backward, warping escaping light into grotesque halos, while the crushing pressure arrests time itself in a silent, suffocating cosmic anvil.*
**Label:** `gemini-3.8-flash`

## 2. Motivation

Knowing which model wrote a text helps check whether a service runs the model it claims and trace where generated content came from. It is hard because the passages are short, all models answer the same prompts, and models from the same lab sound alike. Guessing gives only 5.6%.

## 3. Data

**Source:** [Name That LLM](https://www.kaggle.com/competitions/name-that-llm) on Kaggle, `train.csv` only (the test labels are hidden): 2,699 passages, 150 prompts, 150 passages per model (149 for `claude-fable-5`). License: CC BY-SA 4.0. The data is not committed.

**Split:** by `prompt_id` with a fixed seed (`nll_data.py`): 100 / 25 / 25 prompts = 1,799 / 450 / 450 passages. All answers to a prompt stay in one split, so the model is always tested on new prompts. I choose on validation and use test once.

| Model | Lab | Train | Validation | Test |
|---|---|---:|---:|---:|
| claude-fable-5 | Anthropic | 99 | 25 | 25 |
| claude-opus-4-6 | Anthropic | 100 | 25 | 25 |
| claude-opus-4-8 | Anthropic | 100 | 25 | 25 |
| claude-opus-5 | Anthropic | 100 | 25 | 25 |
| claude-sonnet-5 | Anthropic | 100 | 25 | 25 |
| gemini-3.1-pro-preview | Google | 100 | 25 | 25 |
| gemini-3.6-flash | Google | 100 | 25 | 25 |
| gemini-3.7-flash | Google | 100 | 25 | 25 |
| gemini-3.8-flash | Google | 100 | 25 | 25 |
| kimi-k3 | Moonshot AI | 100 | 25 | 25 |
| gpt-5.5 | OpenAI | 100 | 25 | 25 |
| gpt-5.6-luna | OpenAI | 100 | 25 | 25 |
| gpt-5.6-sol | OpenAI | 100 | 25 | 25 |
| gpt-5.6-terra | OpenAI | 100 | 25 | 25 |
| gpt-6-astra | OpenAI | 100 | 25 | 25 |
| qwen3.8-max | Qwen | 100 | 25 | 25 |
| glm-5.3 | ZAI | 100 | 25 | 25 |
| glm-5.3-flash | ZAI | 100 | 25 | 25 |
| **Total** | | 1,799 | 450 | 450 |

Samples: [samples.md](samples.md). **What I noticed:** every passage is one paragraph with no formatting, so formatting can't give a model away. Punctuation can: Claude and GLM use em dashes (—) in most passages, Gemini and Qwen almost never.

## 4. Method

**Level 0** (from `results/level0_name_that_llm_*.json`):

| Model | Validation balanced accuracy | Test balanced accuracy |
|---|---:|---:|
| TF-IDF character n-grams + logistic regression | 0.264 | 0.298 |

It names the right lab 64.2% of the time on validation, but the right model only 26.4%: telling models from the same lab apart is the hard part.

**Plan:**

| Level | Model | Expected validation balanced accuracy |
|---|---|---:|
| 1 | MLP on TF-IDF features, compared with a BiLSTM | 0.30 |
| 2 | Fine-tuned DistilRoBERTa | 0.38 |
| 3 | Level 3 question, below | – |

**Level 3 question:** Can models from the same lab be told apart, and what gives them away: punctuation, word choice or word order? I will compare model-level and lab-level accuracy, check whether errors stay within a lab, and measure how much accuracy drops when I remove punctuation or shuffle word order.

## 5. Evaluation

Main metric: **balanced accuracy**, because every model should count equally and it is the competition's metric. I also report accuracy, macro-F1 and the confusion matrix. Choices are made on validation; test is used once. The test set is small, so I run 3 seeds. Success means clearly beating Level 0 and answering the Level 3 question with evidence.

## 6. Plan

| Week | What I will do |
|---|---|
| 11/2 – 11/8 | Split, data check, Level 0, proposal |
| 11/9 – 11/15 | Level 1: MLP and BiLSTM |
| 11/16 – 11/22 | Finish Level 1 with 3 seeds (progress report 11/19) |
| 11/23 – 11/29 | Level 2: fine-tune DistilRoBERTa |
| 11/30 – 12/3 | Level 3 tests; presentation (Level 2) |

## Results

| Level | Model | Validation | Test | Results file |
|---|---|---:|---:|---|
| 0 | TF-IDF char n-grams + logistic regression | 0.264 | 0.298 | `results/level0_name_that_llm_val.json`, `_test.json` |

## How to run

```
pip install numpy pandas scikit-learn kaggle
kaggle competitions download -c name-that-llm -p data
unzip data/name-that-llm.zip -d data
python data_check.py
python level0_name_that_llm.py
```

## Data source

Name That LLM, Kaggle: https://www.kaggle.com/competitions/name-that-llm (CC BY-SA 4.0). Not included in this repository.
