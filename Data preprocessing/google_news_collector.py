import requests
import feedparser
import json
import time
import hashlib
from pathlib import Path
from urllib.parse import quote_plus
from datetime import datetime, timezone

OUTPUT_FILE = "Data preprocessing/Data/google_news.json"

REQUEST_DELAY = 2
REQUEST_TIMEOUT = 20

QUERIES = {
    "Sci/Tech": [
        "artificial intelligence",
        "machine learning",
        "generative AI",
        "OpenAI",
        "ChatGPT",
        "Google AI",
        "Microsoft AI",
        "Nvidia",
        "semiconductors",
        "robotics",
        "cybersecurity",
        "quantum computing",
        "space technology",
        "technology startups",
        "software industry"
    ],
    "Business": [
        "stock market",
        "global economy",
        "inflation",
        "interest rates",
        "Federal Reserve",
        "European Central Bank",
        "banking",
        "financial markets",
        "business startups",
        "venture capital",
        "IPO",
        "earnings",
        "oil prices",
        "gold prices",
        "cryptocurrency"
    ],
    "Sports": [
        "football",
        "soccer",
        "Premier League",
        "Champions League",
        "FIFA",
        "NBA",
        "basketball",
        "NFL",
        "tennis",
        "Formula 1",
        "Olympics",
        "golf",
        "boxing",
        "UFC",
        "cricket"
    ],
    "World": [
        "world news",
        "international politics",
        "United Nations",
        "United States politics",
        "European politics",
        "Middle East",
        "Africa",
        "Asia",
        "Russia Ukraine",
        "Israel Palestine",
        "China",
        "India",
        "elections",
        "diplomacy",
        "international conflict"
    ]
}

def clean_text(text):
    if not text:
        return ""
    return " ".join(str(text).split())

def parse_date(entry):
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")

    if parsed:
        try:
            dt = datetime(*parsed[:6], tzinfo=timezone.utc)
            return dt.isoformat()
        except Exception:
            pass

    return entry.get("published") or entry.get("updated") or ""

def create_document(category, entry):
    title = clean_text(entry.get("title", ""))
    summary = clean_text(entry.get("summary", ""))
    url = entry.get("link", "")
    date = parse_date(entry)

    raw_id = f"{title}|{url}"
    document_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]

    source = ""

    if entry.get("source"):
        source = clean_text(entry.get("source", {}).get("title", ""))

    if not source:
        source = "Google News"

    return {
        "id": document_id,
        "source": source,
        "title": title,
        "content": summary,
        "date": date,
        "url": url,
        "category_hint": category
    }

def build_feed_url(query):
    encoded_query = quote_plus(query)
    return (
        f"https://news.google.com/rss/search?"
        f"q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    )

def collect_feed(category, query):
    url = build_feed_url(query)

    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={
                "User-Agent": "TrendTales-NewsCollector/1.0"
            }
        )

        response.raise_for_status()

        feed = feedparser.parse(response.content)

        documents = []

        for entry in feed.entries:
            document = create_document(category, entry)

            if document["title"] and document["url"]:
                documents.append(document)

        print(
            f"  [{category}] {query:<30} -> "
            f"{len(documents)} articles"
        )

        return documents

    except requests.exceptions.RequestException as e:
        print(f"  [{category}] {query:<30} -> ERROR: {e}")
        return []

def deduplicate(documents):
    unique_documents = []
    seen_urls = set()
    seen_titles = set()

    for document in documents:
        url = document.get("url", "").strip()
        title = document.get("title", "").strip().lower()

        if not url or not title:
            continue

        normalized_url = url.split("&oc=")[0]

        if normalized_url in seen_urls:
            continue

        if title in seen_titles:
            continue

        seen_urls.add(normalized_url)
        seen_titles.add(title)

        document["url"] = normalized_url
        unique_documents.append(document)

    return unique_documents

def assign_ids(documents):
    for index, document in enumerate(documents, start=1):
        document["id"] = f"google_news_{index:05d}"

def save_documents(documents):
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            documents,
            file,
            ensure_ascii=False,
            indent=4
        )

def print_summary(documents):
    print("\n" + "=" * 70)
    print("CATEGORY SUMMARY")
    print("=" * 70)

    categories = {}

    for document in documents:
        category = document["category_hint"]
        categories[category] = categories.get(category, 0) + 1

    for category, count in categories.items():
        print(f"{category:<12}: {count}")

    print("=" * 70)
    print(f"Total unique articles: {len(documents)}")
    print("=" * 70)

def main():
    all_documents = []

    total_queries = sum(len(queries) for queries in QUERIES.values())
    completed = 0

    print("=" * 70)
    print("Google News RSS Collector")
    print("=" * 70)
    print(f"Total queries: {total_queries}")
    print()

    for category, queries in QUERIES.items():
        print(f"\n[{category}]")

        for query in queries:
            documents = collect_feed(category, query)

            all_documents.extend(documents)

            completed += 1

            print(
                f"  Progress: {completed}/{total_queries}"
            )

            time.sleep(REQUEST_DELAY)

    print("\nRemoving duplicates...")

    unique_documents = deduplicate(all_documents)

    assign_ids(unique_documents)

    save_documents(unique_documents)

    print_summary(unique_documents)

    print(f"\nSaved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()