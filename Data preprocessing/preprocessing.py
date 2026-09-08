import json
import re
from pathlib import Path


def load_documents(filename="Data preprocessing/Data/sample_data.json"):
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)


def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_documents(documents):
    cleaned_documents = []
    seen = set()

    for document in documents:
        title = document.get("title") or ""
        content = document.get("content") or ""

        title = clean_text(title)
        content = clean_text(content)

        key = (title.lower(), content.lower())

        if key in seen:
            continue

        seen.add(key)

        document["title"] = title
        document["content"] = f"{title}. {content}".strip()

        cleaned_documents.append(document)

    return cleaned_documents

def save_documents(documents, filename="Data preprocessing/Data/cleaned_data.json"):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)  # Fix: auto-create Data/ dir
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(documents, file, ensure_ascii=False, indent=4)


def main():
    documents = load_documents()
    cleaned_documents = preprocess_documents(documents)

    print("Original documents:", len(documents))
    print("Cleaned documents:", len(cleaned_documents))

    save_documents(cleaned_documents)

    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()