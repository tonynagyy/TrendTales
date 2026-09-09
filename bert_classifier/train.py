import os
import glob
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
from datasets import Dataset
# pyrefly: ignore [missing-import]
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import f1_score, precision_score, recall_score
from config import (
    MODEL_NAME, NUM_LABELS, LABEL_MAP,
    MULTI_LABEL_THRESHOLD, TRAIN_SUBSET_SIZE, TEST_SUBSET_SIZE,
    MAX_LENGTH, EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    MODEL_SAVE_PATH, RESULTS_SAVE_PATH, DATA_DIR
)


def load_kaggle_dataset():
    """
    Auto-detects and loads CSV or JSON files inside bert_classifier/data/.
    Converts multi-label news dataset into text and multi-hot float label vectors.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    files = glob.glob(os.path.join(DATA_DIR, "*.csv")) + glob.glob(os.path.join(DATA_DIR, "*.json"))

    if not files:
        raise FileNotFoundError(
            f"No dataset file found in '{DATA_DIR}/'!\n"
            f"Please place your downloaded Kaggle dataset file (e.g. news_dataset.csv) inside '{DATA_DIR}/'."
        )

    data_file = files[0]
    print(f"Loading dataset from: {data_file}")

    if data_file.endswith(".csv"):
        df = pd.read_csv(data_file)
    else:
        df = pd.read_json(data_file)

    print(f"Total rows in raw file: {len(df):,}")

    # Detect text column
    text_col = None
    for col in ["text", "content", "article", "full_text", "body", "description", "title"]:
        if col in df.columns:
            text_col = col
            break
    
    if "title" in df.columns and text_col != "title":
        df["full_text_combined"] = df["title"].fillna("") + ". " + df[text_col].fillna("")
        text_col = "full_text_combined"

    if text_col is None:
        raise ValueError(f"Could not find a text column in {df.columns.tolist()}")


    labels_matrix = []
    
    label_cols = [LABEL_MAP[i] for i in range(NUM_LABELS)]
    existing_label_cols = [col for col in label_cols if col in df.columns]

    if len(existing_label_cols) > 0:
        for _, row in df.iterrows():
            vec = [float(row[col]) if col in row and pd.notnull(row[col]) else 0.0 for col in label_cols]
            labels_matrix.append(vec)
    else:
        cat_col = None
        for candidate in ["revised_labels", "label_cat_list", "labels", "categories", "category", "label"]:
            if candidate in df.columns:
                cat_col = candidate
                break

        if cat_col is None:
            raise ValueError(f"Could not auto-detect category columns in dataset. Available columns: {df.columns.tolist()}")

        print(f"Using category column: '{cat_col}'")
        for _, row in df.iterrows():
            vec = [0.0] * NUM_LABELS
            # Normalize '&' to 'and' and '_' to space so 'Crime & Law' matches 'Crime_and_Law'
            clean_raw_val = str(row[cat_col]).replace("&", "and").replace("_", " ").lower()
            for idx, label_name in LABEL_MAP.items():
                clean_label = label_name.replace("&", "and").replace("_", " ").lower()
                if clean_label in clean_raw_val:
                    vec[idx] = 1.0
            labels_matrix.append(vec)

    formatted_df = pd.DataFrame({
        "text": df[text_col].astype(str).values,
        "labels": labels_matrix
    })

    # Shuffle
    formatted_df = formatted_df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_df = formatted_df.iloc[:min(TRAIN_SUBSET_SIZE, len(formatted_df) - TEST_SUBSET_SIZE)]
    test_df  = formatted_df.iloc[-min(TEST_SUBSET_SIZE, len(formatted_df) - len(train_df)):]

    return Dataset.from_pandas(train_df), Dataset.from_pandas(test_df)


if __name__ == "__main__":
    print("=" * 50)
    print("BERT Multi-Label News Classifier Training")
    print("=" * 50)

    train_data, test_data = load_kaggle_dataset()

    print(f"  Training samples : {len(train_data):,}")
    print(f"  Test samples     : {len(test_data):,}")

    print("\nLoading tokenizer:", MODEL_NAME)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )

    print(f"Tokenizing datasets (MAX_LENGTH={MAX_LENGTH})...")
    train_data = train_data.map(tokenize, batched=True, batch_size=512)
    test_data  = test_data.map(tokenize, batched=True, batch_size=512)

    train_data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_data.set_format("torch",  columns=["input_ids", "attention_mask", "labels"])

    print("\nLoading BERT model with num_labels =", NUM_LABELS)
    # BertForSequenceClassification automatically uses BCEWithLogitsLoss when float labels are passed
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification"
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))  # Sigmoid
        preds = (probs >= MULTI_LABEL_THRESHOLD).astype(int)

        return {
            "f1_macro":    f1_score(labels, preds, average="macro", zero_division=0),
            "f1_micro":    f1_score(labels, preds, average="micro", zero_division=0),
            "precision":   precision_score(labels, preds, average="macro", zero_division=0),
            "recall":      recall_score(labels, preds, average="macro", zero_division=0),
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
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    print("\nStarting Multi-Label Training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    print("\nSaving fine-tuned model...")
    trainer.save_model(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)

    print(f"  Model successfully saved to: {MODEL_SAVE_PATH}/")
    print("\nMulti-Label Training Complete!")
    print("  Next: run python bert_classifier/evaluate.py")
    print("  Then: run python bert_classifier/classify.py")
