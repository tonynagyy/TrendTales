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
from config import MODEL_SAVE_PATH, LABEL_MAP, MAX_LENGTH

_tokenizer = None
_model     = None
_device    = None

def _load_model():
    global _tokenizer, _model, _device

    if _model is not None:
        return  # Already loaded — skip

    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_SAVE_PATH}'.\n"
            f"Please run: python bert_classifier/train.py first."
        )

    print(f"Loading fine-tuned BERT from: {MODEL_SAVE_PATH}")
    _tokenizer = BertTokenizer.from_pretrained(MODEL_SAVE_PATH)
    _model     = BertForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
    _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model.to(_device)
    _model.eval()
    print(f"  Model loaded on: {_device}")


def classify_article(text: str) -> dict:

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


    probabilities = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    predicted_label = int(np.argmax(probabilities))
    confidence      = float(probabilities[predicted_label])

    return {
        "label":      predicted_label,
        "category":   LABEL_MAP[predicted_label],
        "confidence": round(confidence, 4)
    }


#classify_documents (batch — for pipeline integration)
def classify_documents(documents: list, batch_size: int = 32) -> list:

    _load_model()

    print(f"Classifying {len(documents)} documents...")

    texts = [
        f"{doc.get('title', '')}. {doc.get('content', '')}".strip()
        for doc in documents
    ]

    all_labels  = []
    all_confs   = []

    # Process in batches for speed
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

        probs  = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        labels = np.argmax(probs, axis=1)
        confs  = np.max(probs, axis=1)

        all_labels.extend(labels.tolist())
        all_confs.extend(confs.tolist())

        if (i // batch_size) % 5 == 0:
            print(f"  Progress: {min(i + batch_size, len(texts))}/{len(texts)}")

    # Add classification results back into each document dict
    classified_documents = []
    for doc, label, conf in zip(documents, all_labels, all_confs):
        doc_copy = dict(doc) 
        doc_copy["category"]   = LABEL_MAP[label]
        doc_copy["confidence"] = round(float(conf), 4)
        classified_documents.append(doc_copy)

    from collections import Counter
    cat_counts = Counter(d["category"] for d in classified_documents)
    print("\nClassification summary:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat:12s}: {count:3d} articles")

    return classified_documents


def save_classified_documents(documents: list, path: str = "Data preprocessing/Data/classified_data.json"):
    """Save classified documents to JSON for use by Person 3 (RAG)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=4)
    print(f"\nClassified documents saved to: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing BERT classifier with sample articles")
    print("=" * 60)

    test_articles = [
        "OpenAI launches GPT-5 with advanced reasoning capabilities and multimodal features",
        "Manchester United beats Arsenal 3-1 in Premier League opener at Old Trafford",
        "Federal Reserve holds interest rates steady amid inflation concerns in September",
        "UN Security Council meets to discuss ceasefire negotiations in ongoing conflict",
    ]

    expected = ["Sci/Tech", "Sports", "Business", "World"]

    print("\nSingle article classification test:")
    correct = 0
    for text, exp in zip(test_articles, expected):
        result = classify_article(text)
        status = "✅" if result["category"] == exp else "❌"
        print(f"\n  {status} Input    : {text[:60]}...")
        print(f"     Predicted : {result['category']} (confidence: {result['confidence']:.2%})")
        print(f"     Expected  : {exp}")
        if result["category"] == exp:
            correct += 1

    print(f"\n  Test accuracy: {correct}/{len(test_articles)} = {correct/len(test_articles):.0%}")

    # Test batch classification on Person 1's data if it exists
    cleaned_data_path = "Data preprocessing/Data/cleaned_data.json"
    if os.path.exists(cleaned_data_path):
        print("\n" + "=" * 60)
        print("Batch classification on Person 1's data:")
        print("=" * 60)
        with open(cleaned_data_path, "r", encoding="utf-8") as f:
            documents = json.load(f)

        classified = classify_documents(documents)
        save_classified_documents(classified)
    else:
        print(f"\n  Note: {cleaned_data_path} not found — skipping batch test.")
        print("  Run Person 1's pipeline first to generate the data.")
