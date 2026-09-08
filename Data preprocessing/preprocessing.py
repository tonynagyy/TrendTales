import json
import re
from pathlib import Path
from html import unescape


def load_documents(filename="Data preprocessing/Data/final_news.json"):
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)


def clean_text(text):
    text = text or ""
    text = unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\[\+\d+\s+chars\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_documents(documents):
    cleaned_documents = []
    seen = set()

    for document in documents:
        title = clean_text(document.get("title"))
        content = clean_text(document.get("content"))

        key = (title.lower(), content.lower())

        if key in seen:
            continue

        seen.add(key)

        document["title"] = title
        document["content"] = f"{title}. {content}".strip()

        cleaned_documents.append(document)

    return cleaned_documents


def save_documents(
    documents,
    filename="Data preprocessing/Data/cleaned_data.json"
):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(
            documents,
            file,
            ensure_ascii=False,
            indent=4
        )


def main():
    documents = load_documents()
    cleaned_documents = preprocess_documents(documents)

    print("Original documents:", len(documents))
    print("Cleaned documents:", len(cleaned_documents))

    save_documents(cleaned_documents)

    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()
