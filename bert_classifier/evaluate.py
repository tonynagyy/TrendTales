import os
import json
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
# pyrefly: ignore [missing-import]
from datasets import load_dataset
# pyrefly: ignore [missing-import]
from transformers import BertTokenizer, BertForSequenceClassification
# pyrefly: ignore [missing-import]
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from config import (
    MODEL_NAME, DATASET_NAME, NUM_LABELS, LABEL_MAP,
    MAX_LENGTH, BATCH_SIZE, TEST_SUBSET_SIZE,
    MODEL_SAVE_PATH, RESULTS_SAVE_PATH
)

os.makedirs(RESULTS_SAVE_PATH, exist_ok=True)


print("Loading fine-tuned model from:", MODEL_SAVE_PATH)

tokenizer = BertTokenizer.from_pretrained(MODEL_SAVE_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"  Running on: {device}")


print("\nLoading test data...")
dataset = load_dataset(DATASET_NAME)
test_data = dataset["test"].shuffle(seed=42).select(range(TEST_SUBSET_SIZE))
print(f"  Test samples: {len(test_data):,}")


print("\nRunning predictions...")

all_preds  = []
all_labels = []

for i in range(0, len(test_data), BATCH_SIZE):
    batch_texts  = test_data[i:i + BATCH_SIZE]["text"]
    batch_labels = test_data[i:i + BATCH_SIZE]["label"]

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

    logits = outputs.logits.cpu().numpy()
    preds  = np.argmax(logits, axis=1)

    all_preds.extend(preds)
    all_labels.extend(batch_labels)

    if (i // BATCH_SIZE) % 10 == 0:
        print(f"  Processed {min(i + BATCH_SIZE, len(test_data))}/{len(test_data)} samples...")

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

label_names = [LABEL_MAP[i] for i in range(NUM_LABELS)]


#   Precision = Of all articles we labeled "Sports", how many were actually Sports?
#   Recall    = Of all actual Sports articles, how many did we correctly find?
#   F1        = Harmonic mean of Precision and Recall (single balanced score)
#   Support   = Number of test samples in that class
print("\n" + "=" * 40)
print("CLASSIFICATION REPORT")
print("=" * 40)

report = classification_report(all_labels, all_preds, target_names=label_names, digits=4)
accuracy = accuracy_score(all_labels, all_preds)

print(report)
print(f"Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

report_path = f"{RESULTS_SAVE_PATH}/classification_report.txt"
with open(report_path, "w") as f:
    f.write("BERT News Classifier — Evaluation Report\n")
    f.write("=" * 40 + "\n\n")
    f.write(report)
    f.write(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
print(f"\nReport saved to: {report_path}")

print("\nGenerating confusion matrix...")

cm = confusion_matrix(all_labels, all_preds)
cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # Row-normalize

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(
    cm, annot=True, fmt="d",
    xticklabels=label_names, yticklabels=label_names,
    cmap="Blues", ax=axes[0]
)
axes[0].set_title("Confusion Matrix (Raw Counts)", fontsize=13, fontweight="bold")
axes[0].set_ylabel("True Label", fontsize=11)
axes[0].set_xlabel("Predicted Label", fontsize=11)

sns.heatmap(
    cm_normalized, annot=True, fmt=".2%",
    xticklabels=label_names, yticklabels=label_names,
    cmap="Greens", ax=axes[1]
)
axes[1].set_title("Confusion Matrix (Normalized)", fontsize=13, fontweight="bold")
axes[1].set_ylabel("True Label", fontsize=11)
axes[1].set_xlabel("Predicted Label", fontsize=11)

plt.suptitle(f"BERT Fine-Tuned News Classifier — Accuracy: {accuracy:.2%}", fontsize=14)
plt.tight_layout()

cm_path = f"{RESULTS_SAVE_PATH}/confusion_matrix.png"
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Confusion matrix saved to: {cm_path}")

print("\nGenerating per-class F1 bar chart...")

from sklearn.metrics import f1_score, precision_score, recall_score

f1_per_class        = f1_score(all_labels, all_preds, average=None)
precision_per_class = precision_score(all_labels, all_preds, average=None)
recall_per_class    = recall_score(all_labels, all_preds, average=None)

x = np.arange(len(label_names))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, precision_per_class, width, label="Precision", color="#4C72B0")
ax.bar(x,         f1_per_class,        width, label="F1 Score",  color="#55A868")
ax.bar(x + width, recall_per_class,    width, label="Recall",    color="#C44E52")

ax.set_xticks(x)
ax.set_xticklabels(label_names, fontsize=12)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Per-Class Precision, F1, and Recall\nFine-Tuned BERT on AG News", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.axhline(y=accuracy, color="gray", linestyle="--", alpha=0.7, label=f"Avg Accuracy: {accuracy:.2%}")

for i, (p, f, r) in enumerate(zip(precision_per_class, f1_per_class, recall_per_class)):
    ax.text(i - width, p + 0.01, f"{p:.2f}", ha="center", va="bottom", fontsize=9)
    ax.text(i,         f + 0.01, f"{f:.2f}", ha="center", va="bottom", fontsize=9)
    ax.text(i + width, r + 0.01, f"{r:.2f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
bar_path = f"{RESULTS_SAVE_PATH}/per_class_metrics.png"
plt.savefig(bar_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Per-class chart saved to: {bar_path}")

print("\n Evaluation complete!")
print(f"   All results in: {RESULTS_SAVE_PATH}/")
