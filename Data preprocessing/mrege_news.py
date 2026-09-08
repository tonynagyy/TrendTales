import json
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode


BASE_DIR = Path("Data preprocessing/Data")

INPUT_FILES = [
    BASE_DIR / "sample_data.json",
    BASE_DIR / "rss_news.json",
    BASE_DIR / "google_news.json"
]

OUTPUT_FILE = BASE_DIR / "final_news.json"


def load_json_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_url(url):
    if not url:
        return ""

    try:
        parts = urlsplit(url.strip())

        query_params = [
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if key.lower() != "oc"
        ]

        return urlunsplit((
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path.rstrip("/"),
            urlencode(query_params),
            ""
        ))
    except Exception:
        return url.strip().lower()


def normalize_title(title):
    title = title or ""
    title = title.lower()
    title = re.sub(r"\s+", " ", title)
    title = re.sub(r"[^\w\s]", "", title)
    return title.strip()


def clean_document(document):
    return {
        "source": document.get("source") or "Unknown",
        "title": document.get("title") or "",
        "content": document.get("content") or "",
        "date": document.get("date") or "",
        "url": document.get("url") or "",
        "category_hint": document.get("category_hint") or ""
    }


def merge_documents():
    all_documents = []
    loaded_files = {}

    for filename in INPUT_FILES:
        if not filename.exists():
            print(f"Skipped: {filename}")
            continue

        documents = load_json_file(filename)
        loaded_files[str(filename)] = len(documents)
        all_documents.extend(documents)

    print("\nLoaded files:")
    for filename, count in loaded_files.items():
        print(f"{filename}: {count}")

    print(f"\nTotal before merge: {len(all_documents)}")

    merged_documents = []
    seen_urls = set()
    seen_titles = set()

    duplicates_by_url = 0
    duplicates_by_title = 0
    skipped_invalid = 0

    for document in all_documents:
        cleaned = clean_document(document)

        title = cleaned["title"].strip()
        content = cleaned["content"].strip()
        url = cleaned["url"].strip()

        if not title and not content:
            skipped_invalid += 1
            continue

        normalized_url = normalize_url(url)
        normalized_title = normalize_title(title)

        if normalized_url:
            if normalized_url in seen_urls:
                duplicates_by_url += 1
                continue
            seen_urls.add(normalized_url)

        if normalized_title:
            if normalized_title in seen_titles:
                duplicates_by_title += 1
                continue
            seen_titles.add(normalized_title)

        merged_documents.append(cleaned)

    for index, document in enumerate(merged_documents, start=1):
        document["id"] = f"news_{index:05d}"

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(merged_documents, file, ensure_ascii=False, indent=4)

    print(f"\nDuplicates by URL: {duplicates_by_url}")
    print(f"Duplicates by title: {duplicates_by_title}")
    print(f"Skipped invalid: {skipped_invalid}")
    print(f"Final unique articles: {len(merged_documents)}")
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    merge_documents()

