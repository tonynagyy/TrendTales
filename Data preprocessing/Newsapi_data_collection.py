import os
import requests
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # Loads .env file automatically


TOPIC_QUERIES = {
    "Sci/Tech":     "artificial intelligence OR machine learning OR openai OR nvidia OR chatgpt OR robotics",
    "Business":     "stock market OR economy OR inflation OR startup OR IPO OR earnings OR Federal Reserve",
    "Sports":       "football OR basketball OR tennis OR FIFA OR Olympics OR NBA OR Premier League",
    "World":        "election OR war OR diplomacy OR climate OR United Nations OR government OR conflict",
}

PAGE_SIZE = 100  


def create_document(id, source, title, content, date, category_hint=""):
    return {
        "id": id,
        "source": source,
        "title": title,
        "content": content,
        "date": date,
        "category_hint": category_hint   
    }


def collect_news_for_topic(topic_label, query, page_size=PAGE_SIZE):
    """Fetch up to page_size articles for a single topic query."""
    api_key = os.getenv("NEWS_API_KEY") # Make sure to set your NEWS_API_KEY in the .env file or use directly in the code (not recomended)

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size
    }

    headers = {
        "X-Api-Key": api_key
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()
        articles = data.get("articles", [])

        documents = []
        for article in articles:
            content = article.get("description") or article.get("content") or ""

            document = create_document(
                id="",                              
                source=article["source"]["name"],
                title=article.get("title") or "",
                content=content,
                date=article["publishedAt"],
                category_hint=topic_label
            )
            documents.append(document)

        print(f"  [{topic_label}] Collected {len(documents)} articles")
        return documents

    except requests.exceptions.HTTPError as e:
        print(f"  [{topic_label}] HTTP error: {e}")
        return []
    except requests.exceptions.RequestException as e:
        print(f"  [{topic_label}] Network error: {e}")
        return []


def collect_all_news(topics=TOPIC_QUERIES, page_size=PAGE_SIZE):

    all_documents = []

    print(f"Collecting news from {len(topics)} categories...")

    for i, (label, query) in enumerate(topics.items()):
        docs = collect_news_for_topic(label, query, page_size)
        all_documents.extend(docs)

        if i < len(topics) - 1:
            time.sleep(1)

    for i, doc in enumerate(all_documents):
        doc["id"] = f"news_{i+1:04d}"

    print(f"\nTotal articles collected: {len(all_documents)}")
    return all_documents


def save_documents(documents, filename="Data preprocessing/Data/sample_data.json"):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(documents, file, ensure_ascii=False, indent=4)


def main():
    documents = collect_all_news()
    save_documents(documents)
    print("Data saved successfully.")


if __name__ == "__main__":
    main()