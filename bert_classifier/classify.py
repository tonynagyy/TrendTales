import os
import sys
import json
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import numpy as np
from pathlib import Path
# pyrefly: ignore [missing-import]
from transformers import BertTokenizer, BertForSequenceClassification

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MODEL_SAVE_PATH, LABEL_MAP,
    MAX_LENGTH, MULTI_LABEL_THRESHOLD,
    INPUT_DATA_PATH, OUTPUT_DATA_PATH
)

_tokenizer = None
_model     = None
_device    = None

def _load_model():
    global _tokenizer, _model, _device

    if _model is not None:
        return

    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_SAVE_PATH}'.\n"
            f"Please run: python bert_classifier/train.py first."
        )

    print(f"Loading fine-tuned Multi-Label BERT from: {MODEL_SAVE_PATH}")
    _tokenizer = BertTokenizer.from_pretrained(MODEL_SAVE_PATH)
    _model     = BertForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
    _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model.to(_device)
    _model.eval()
    print(f"  Loaded on: {_device}")


def classify_article(text: str, threshold: float = MULTI_LABEL_THRESHOLD) -> dict:
    """
    Classifies a single article into one or more categories using Sigmoid probabilities.
    """
    _load_model()

    inputs = _tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)

    # Sigmoid for multi-label probabilities
    probabilities = torch.sigmoid(outputs.logits).cpu().numpy()[0]

    assigned_categories = []
    confidence_map = {}

    for idx, prob in enumerate(probabilities):
        cat_name = LABEL_MAP[idx]
        confidence_map[cat_name] = round(float(prob), 4)
        if prob >= threshold:
            assigned_categories.append(cat_name)

    # Fallback: if no label exceeded threshold, pick highest argmax
    if not assigned_categories:
        top_idx = int(np.argmax(probabilities))
        assigned_categories = [LABEL_MAP[top_idx]]

    return {
        "categories": assigned_categories,
        "primary_category": assigned_categories[0],
        "confidence": confidence_map
    }


def classify_documents(documents: list, batch_size: int = 32, threshold: float = MULTI_LABEL_THRESHOLD) -> list:
    """
    Batch classifies a list of news document dicts and adds 'categories' & 'confidence' to each.
    """
    _load_model()

    print(f"Batch classifying {len(documents)} documents...")

    texts = [
        f"{doc.get('title', '')}. {doc.get('content', '')}".strip()
        for doc in documents
    ]

    all_categories = []
    all_confidences = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        inputs = _tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        inputs = {k: v.to(_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = _model(**inputs)

        probs_batch = torch.sigmoid(outputs.logits).cpu().numpy()

        for probs in probs_batch:
            assigned = []
            conf_map = {}
            for idx, prob in enumerate(probs):
                cat_name = LABEL_MAP[idx]
                conf_map[cat_name] = round(float(prob), 4)
                if prob >= threshold:
                    assigned.append(cat_name)

            if not assigned:
                top_idx = int(np.argmax(probs))
                assigned = [LABEL_MAP[top_idx]]

            all_categories.append(assigned)
            all_confidences.append(conf_map)

        if (i // batch_size) % 5 == 0:
            print(f"  Progress: {min(i + batch_size, len(texts))}/{len(texts)}")

    # Update document dicts
    classified_documents = []
    for doc, cats, conf_map in zip(documents, all_categories, all_confidences):
        doc_copy = dict(doc)
        doc_copy["category"]    = cats[0]          # Primary category for backward compatibility
        doc_copy["categories"]  = cats             # Full multi-label list
        doc_copy["confidence"]  = conf_map[cats[0]]# Primary confidence score
        doc_copy["all_confidences"] = conf_map
        classified_documents.append(doc_copy)

    print("\nBatch classification complete.")
    return classified_documents


def save_classified_documents(documents: list, path: str = OUTPUT_DATA_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=4)
    print(f"Classified documents saved to: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Multi-Label BERT Classifier")
    print("=" * 60)

    test_article = "Apple announces acquisition of AI startup to boost autonomous driving technology"
    result = classify_article(test_article)

    print(f"\nTest text: '{test_article}'")
    print(f"Assigned Categories: {result['categories']}")
    print(f"Primary Category   : {result['primary_category']}")
    print("Top Confidences    :")
    for cat, score in sorted(result["confidence"].items(), key=lambda x: x[1], reverse=True)[:4]:
        print(f"  {cat:22s}: {score:.2%}")

    if os.path.exists(INPUT_DATA_PATH):
        print("\n" + "=" * 60)
        print(f"Classifying real news pipeline data ({INPUT_DATA_PATH}):")
        print("=" * 60)
        with open(INPUT_DATA_PATH, "r", encoding="utf-8") as f:
            documents = json.load(f)

        classified = classify_documents(documents)
        save_classified_documents(classified)
    else:
        print(f"\nNote: {INPUT_DATA_PATH} not found yet. Run Person 1's pipeline when ready.")
