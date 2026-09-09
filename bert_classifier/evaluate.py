import os
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score
)
# pyrefly: ignore [missing-import]
from transformers import BertTokenizer, BertForSequenceClassification
from config import (
    NUM_LABELS, LABEL_MAP, MULTI_LABEL_THRESHOLD,
    MAX_LENGTH, BATCH_SIZE, MODEL_SAVE_PATH, RESULTS_SAVE_PATH
)
from train import load_kaggle_dataset

os.makedirs(RESULTS_SAVE_PATH, exist_ok=True)

print("Loading fine-tuned multi-label BERT model from:", MODEL_SAVE_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_SAVE_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"  Device: {device}")

print("\nLoading test dataset...")
_, test_data = load_kaggle_dataset()
print(f"  Test samples: {len(test_data):,}")

print("\nRunning multi-label evaluation...")

all_probs = []
all_labels = []

for i in range(0, len(test_data), BATCH_SIZE):
    batch = test_data[i:i + BATCH_SIZE]
    batch_texts = batch["text"]
    batch_target_labels = np.array(batch["labels"])

    inputs = tokenizer(
        batch_texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Sigmoid for multi-label probabilities
    probs = torch.sigmoid(outputs.logits).cpu().numpy()

    all_probs.extend(probs)
    all_labels.extend(batch_target_labels)

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

# Apply threshold
all_preds = (all_probs >= MULTI_LABEL_THRESHOLD).astype(int)

label_names = [LABEL_MAP[i] for i in range(NUM_LABELS)]

print("\n" + "=" * 50)
print("MULTI-LABEL CLASSIFICATION REPORT")
print("=" * 50)

report = classification_report(
    all_labels,
    all_preds,
    target_names=label_names,
    digits=4,
    zero_division=0
)

f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
f1_micro = f1_score(all_labels, all_preds, average="micro", zero_division=0)

print(report)
print(f"F1 Score (Macro): {f1_macro:.4f}")
print(f"F1 Score (Micro): {f1_micro:.4f}")

report_path = os.path.join(RESULTS_SAVE_PATH, "classification_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("BERT Multi-Label News Classifier — Evaluation Report\n")
    f.write("=" * 50 + "\n\n")
    f.write(report)
    f.write(f"\nF1 Macro: {f1_macro:.4f}\nF1 Micro: {f1_micro:.4f}\n")
print(f"\nReport saved to: {report_path}")

print("\nGenerating per-class metrics chart...")

f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)

x = np.arange(len(label_names))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - width, precision_per_class, width, label="Precision", color="#4C72B0")
ax.bar(x,         f1_per_class,        width, label="F1 Score",  color="#55A868")
ax.bar(x + width, recall_per_class,    width, label="Recall",    color="#C44E52")

ax.set_xticks(x)
ax.set_xticklabels(label_names, rotation=35, ha="right", fontsize=10)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Per-Category Precision, F1, and Recall (Multi-Label BERT)", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)

plt.tight_layout()
bar_path = os.path.join(RESULTS_SAVE_PATH, "per_class_metrics.png")
plt.savefig(bar_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Per-class chart saved to: {bar_path}")

print("\nMulti-Label Evaluation Complete!")
