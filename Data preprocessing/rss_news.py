import json
import time
import hashlib
import requests
import feedparser
from pathlib import Path
from datetime import datetime, timezone


RSS_FEEDS = {
    "Sci/Tech": {
        "BBC Technology": "https://feeds.bbci.co.uk/news/technology/rss.xml",
        "BBC Science": "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
        "TechCrunch": "https://techcrunch.com/feed/",
        "The Verge": "https://www.theverge.com/rss/index.xml",
        "Ars Technica": "https://feeds.arstechnica.com/arstechnica/index",
        "MIT Technology Review": "https://www.technologyreview.com/feed/",
        "Wired": "https://www.wired.com/feed/rss",
        "ScienceDaily": "https://www.sciencedaily.com/rss/top/science.xml"
    },
    "Business": {
        "BBC Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
        "BBC Economy": "https://feeds.bbci.co.uk/news/business/economy/rss.xml",
        "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
        "The Guardian Business": "https://www.theguardian.com/business/rss",
        "The Guardian Economics": "https://www.theguardian.com/business/economics/rss",
        "Forbes": "https://www.forbes.com/real-time/feed2/",
        "Investing": "https://www.investing.com/rss/news.rss"
    },
    "Sports": {
        "BBC Sport": "https://feeds.bbci.co.uk/sport/rss.xml",
        "BBC Football": "https://feeds.bbci.co.uk/sport/football/rss.xml",
        "BBC Tennis": "https://feeds.bbci.co.uk/sport/tennis/rss.xml",
        "BBC Formula 1": "https://feeds.bbci.co.uk/sport/formula1/rss.xml",
        "BBC Golf": "https://feeds.bbci.co.uk/sport/golf/rss.xml",
        "BBC Boxing": "https://feeds.bbci.co.uk/sport/boxing/rss.xml",
        "ESPN": "https://www.espn.com/espn/rss/news",
        "Sky Sports": "https://www.skysports.com/rss/12040",
        "The Guardian Sport": "https://www.theguardian.com/sport/rss"
    },
    "World": {
        "BBC World": "https://feeds.bbci.co.uk/news/world/rss.xml",
        "BBC Europe": "https://feeds.bbci.co.uk/news/world/europe/rss.xml",
        "BBC Middle East": "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml",
        "BBC Africa": "https://feeds.bbci.co.uk/news/world/africa/rss.xml",
        "BBC Asia": "https://feeds.bbci.co.uk/news/world/asia/rss.xml",
        "BBC US & Canada": "https://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml",
        "NPR World": "https://feeds.npr.org/1004/rss.xml",
        "The Guardian World": "https://www.theguardian.com/world/rss",
        "The Guardian Europe": "https://www.theguardian.com/world/europe-news/rss"
    }
}


REQUEST_DELAY = 2
REQUEST_TIMEOUT = 20


def clean_text(text):
    if not text:
        return ""

    return " ".join(text.replace("\n", " ").split())


def parse_date(entry):
    parsed_time = entry.get("published_parsed") or entry.get("updated_parsed")

    if parsed_time:
        try:
            return datetime(
                *parsed_time[:6],
                tzinfo=timezone.utc
            ).isoformat()
        except (TypeError, ValueError):
            pass

    return entry.get("published") or entry.get("updated") or ""


def create_document(
    source,
    title,
    content,
    date,
    url,
    category_hint
):
    document_id = hashlib.sha256(
        url.encode("utf-8")
    ).hexdigest()[:16]

    return {
        "id": document_id,
        "source": source,
        "title": title,
        "content": content,
        "date": date,
        "url": url,
        "category_hint": category_hint
    }


def collect_feed(
    source_name,
    feed_url,
    category
):
    print(f"\n[{category}] {source_name}")
    print(f"Feed: {feed_url}")

    headers = {
        "User-Agent": "TrendTales-NewsCollector/1.0"
    }

    try:
        response = requests.get(
            feed_url,
            headers=headers,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        feed = feedparser.parse(response.content)

        if not feed.entries:
            print("No entries found.")
            return []

        documents = []

        for entry in feed.entries:
            title = clean_text(
                entry.get("title", "")
            )

            url = entry.get("link", "")

            if not title or not url:
                continue

            content = clean_text(
                entry.get("summary", "")
            )

            if not content:
                content = clean_text(
                    entry.get("description", "")
                )

            document = create_document(
                source=source_name,
                title=title,
                content=content,
                date=parse_date(entry),
                url=url,
                category_hint=category
            )

            documents.append(document)

        print(
            f"Collected: {len(documents)} articles"
        )

        return documents

    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return []

    except Exception as e:
        print(f"Unexpected error: {e}")
        return []


def remove_duplicates(documents):
    unique_documents = []
    seen_urls = set()
    seen_titles = set()

    for document in documents:
        url = document.get("url", "").strip()
        title = document.get("title", "").strip().lower()

        if not url:
            continue

        if url in seen_urls:
            continue

        if title and title in seen_titles:
            continue

        seen_urls.add(url)

        if title:
            seen_titles.add(title)

        unique_documents.append(document)

    return unique_documents


def assign_ids(documents):
    for index, document in enumerate(
        documents,
        start=1
    ):
        document["id"] = f"news_{index:05d}"

    return documents


def collect_all_news():
    all_documents = []

    print("=" * 70)
    print("TrendTales RSS News Collector V2")
    print("=" * 70)

    total_feeds = sum(
        len(feeds)
        for feeds in RSS_FEEDS.values()
    )

    print(f"Total RSS feeds: {total_feeds}")
    print("=" * 70)

    for category, sources in RSS_FEEDS.items():

        print(
            f"\n{'=' * 70}"
        )
        print(
            f"CATEGORY: {category}"
        )
        print(
            f"{'=' * 70}"
        )

        for source_name, feed_url in sources.items():

            documents = collect_feed(
                source_name,
                feed_url,
                category
            )

            all_documents.extend(
                documents
            )

            time.sleep(
                REQUEST_DELAY
            )

    print(
        "\nRemoving duplicates..."
    )

    all_documents = remove_duplicates(
        all_documents
    )

    all_documents = assign_ids(
        all_documents
    )

    print(
        f"\nFinal unique articles: "
        f"{len(all_documents)}"
    )

    return all_documents


def print_summary(documents):
    print(
        "\n" + "=" * 70
    )
    print(
        "COLLECTION SUMMARY"
    )
    print(
        "=" * 70
    )

    category_counts = {}
    source_counts = {}

    for document in documents:

        category = document.get(
            "category_hint",
            ""
        )

        source = document.get(
            "source",
            ""
        )

        category_counts[category] = (
            category_counts.get(
                category,
                0
            ) + 1
        )

        source_counts[source] = (
            source_counts.get(
                source,
                0
            ) + 1
        )

    print("\nBy Category:")

    for category, count in category_counts.items():
        print(
            f"{category}: {count}"
        )

    print("\nBy Source:")

    for source, count in source_counts.items():
        print(
            f"{source}: {count}"
        )

    print(
        f"\nTotal unique articles: "
        f"{len(documents)}"
    )

    print(
        "=" * 70
    )


def save_documents(
    documents,
    filename="Data preprocessing/Data/rss_news.json"
):
    Path(filename).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            documents,
            file,
            ensure_ascii=False,
            indent=4
        )

    print(
        f"\nData saved to: {filename}"
    )


def main():
    documents = collect_all_news()

    save_documents(
        documents
    )

    print_summary(
        documents
    )

    print(
        "\nRSS data collection completed successfully."
    )


if __name__ == "__main__":
    main()