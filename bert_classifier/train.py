import os
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from datasets import load_dataset
# pyrefly: ignore [missing-import]
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score
from config import (
    MODEL_NAME, DATASET_NAME, NUM_LABELS,
    TRAIN_SUBSET_SIZE, TEST_SUBSET_SIZE,
    MAX_LENGTH, EPOCHS, BATCH_SIZE,
    LEARNING_RATE, WEIGHT_DECAY,
    MODEL_SAVE_PATH, RESULTS_SAVE_PATH
)


print("=" * 40)
print("Loading AG News dataset...")
print("=" * 40)

dataset = load_dataset(DATASET_NAME)

train_data = dataset["train"].shuffle(seed=42).select(range(TRAIN_SUBSET_SIZE))
test_data  = dataset["test"].shuffle(seed=42).select(range(TEST_SUBSET_SIZE))

print(f"  Training samples : {len(train_data):,}")
print(f"  Test samples     : {len(test_data):,}")
print(f"  Label distribution (train):")
from collections import Counter
label_counts = Counter(train_data["label"])
label_names = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
for label_id, count in sorted(label_counts.items()):
    print(f"    {label_names[label_id]}: {count:,} samples")


print("\n" + "=" * 40)
print("Loading tokenizer...")
print("=" * 40)

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    """
    Convert raw text into BERT input tensors.

    AG News has a "text" column with: "Title. Description. Body."
    We truncate to MAX_LENGTH=128 which comfortably fits title+description.
    """
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

print(f"  Tokenizing {TRAIN_SUBSET_SIZE:,} training samples...")
train_data = train_data.map(tokenize, batched=True, batch_size=256)
print(f"  Tokenizing {TEST_SUBSET_SIZE:,} test samples...")
test_data  = test_data.map(tokenize, batched=True, batch_size=256)

train_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_data.set_format("torch",  columns=["input_ids", "attention_mask", "label"])

print("Tokenization complete.")


print("\n" + "=" * 40)
print("Loading BERT model...")
print("=" * 40)

model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters     : {total_params:,}")
print(f"  Trainable parameters : {trainable_params:,}")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)  

    return {
        "accuracy":    accuracy_score(labels, predictions),
        "f1_macro":    f1_score(labels, predictions, average="macro"),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
    }


os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(RESULTS_SAVE_PATH, exist_ok=True)

training_args = TrainingArguments(
    output_dir=RESULTS_SAVE_PATH, 
    num_train_epochs=EPOCHS,               
    per_device_train_batch_size=BATCH_SIZE, 
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,            
    weight_decay=WEIGHT_DECAY,              
    eval_strategy="epoch",       
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    logging_steps=50,                       
    warmup_steps=100,
    fp16=True,                             # Enabled for RTX 4050 GPU
    report_to="none",
)


print("\n" + "=" * 40)
print("Starting training...")
print(f"  Model       : {MODEL_NAME}")
print(f"  Dataset     : {DATASET_NAME} ({TRAIN_SUBSET_SIZE:,} train / {TEST_SUBSET_SIZE:,} test)")
print(f"  Epochs      : {EPOCHS}")
print(f"  Batch size  : {BATCH_SIZE}")
print(f"  LR          : {LEARNING_RATE}")
print(f"  Max length  : {MAX_LENGTH} tokens")
print("=" * 40)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

print("\n" + "=" * 40)
print("Saving model...")
print("=" * 40)

trainer.save_model(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

print(f"  Model saved to: {MODEL_SAVE_PATH}/")
print("\n Training complete!")
print("   Next step: run evaluate.py to see detailed metrics")
print("   Then run:  python bert_classifier/classify.py")
