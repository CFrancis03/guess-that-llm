Task: Name That LLM. Predict which of 18 LLMs wrote a passage (Kaggle: name-that-llm).

Rules for any coding agent working in this repository:
- The split is fixed: split by prompt_id in nll_data.py (seed 42, 100/25/25 prompts). Never re-split the data or change the seed.
- Never train on validation or test rows.
- Main metric: balanced accuracy. Also report accuracy, macro-F1 and the confusion matrix.
- Choose models, hyperparameters and epochs on the validation set only. Evaluate each final model on the test set once.
- Every reported number must come from a results file in results/. Save the metrics of every run there.
- Do not commit the Kaggle data (data/train.csv, data/test.csv, the zip) or model weights.
- One training run must take under 30 minutes on a free Colab T4 GPU.
