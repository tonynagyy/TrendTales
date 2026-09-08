
import json
import re
import math
from collections import defaultdict
from datetime import datetime, timezone
from html import unescape

INPUT_FILE = "Data preprocessing/Data/final_news.json"
OUTPUT_FILE = "Data preprocessing/Data/top_trends.json"

RECENT_DAYS = 120
TOP_N = 10
MIN_FREQUENCY = 3

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than",
    "of", "to", "in", "on", "for", "from", "with", "by", "at",
    "as", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "they", "their",
    "them", "he", "she", "his", "her", "you", "your", "we", "our",
    "i", "me", "my", "us", "about", "into", "over", "after",
    "before", "during", "between", "through", "against", "without",
    "can", "could", "would", "should", "will", "may", "might",
    "do", "does", "did", "not", "no", "more", "most", "some",
    "any", "all", "how", "what", "when", "where", "who", "why",
    "which", "while", "also", "just", "only", "very", "many",
    "much", "one", "two", "three", "first", "last",
    "has", "have", "had", "having", "get", "got", "gets",
    "go", "goes", "going", "gone", "come", "comes", "came",
    "make", "makes", "made", "take", "takes", "took", "taken",
    "out", "up", "down", "over", "back", "still", "well",
    "now", "then", "here", "there", "where", "why", "how",
    "new", "said", "says", "say", "according"
}

GENERIC_WORDS = {
    "world", "sports", "sport", "news", "global",
    "league", "prices", "price", "market", "markets",
    "business", "today", "latest", "report", "reports",
    "amid", "live", "top", "breaking", "time", "week",
    "weeks", "month", "months", "year", "years", "people",
    "country", "countries", "international", "update",
    "updates", "newsroom", "story", "stories", "official",
    "officials", "article", "articles", "read", "click",
    "share", "google", "rss", "feed", "feeds", "source",
    "sources", "http", "https", "www", "com", "org", "net",
    "href", "target", "blank", "nbsp", "font", "color",
    "style", "class", "html", "xml", "url", "link", "links",
    "view", "page", "pages", "times"
}

def parse_date(value):
    if not value:
        return None

    value = str(value).strip()

    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except:
        pass

    formats = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z"
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(value, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except:
            continue

    return None

def clean_text(text):
    text = str(text)
    text = unescape(text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\b(?:href|target|class|style)\s*=\s*[\"'][^\"']*[\"']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize(text):
    text = clean_text(text)

    words = re.findall(
        r"\b[a-zA-Z][a-zA-Z0-9-]{2,}\b",
        text.lower()
    )

    return {
        word.strip("-")
        for word in words
        if word not in STOPWORDS
        and word not in GENERIC_WORDS
        and len(word.strip("-")) >= 3
        and not word.isdigit()
    }

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    articles = json.load(f)

now = datetime.now(timezone.utc)

recent_articles = []

for article in articles:
    date = parse_date(article.get("date"))

    if date is None:
        continue

    age = (now - date).total_seconds() / 86400

    if 0 <= age <= RECENT_DAYS:
        article["_age_days"] = age
        recent_articles.append(article)

print(f"Documents loaded: {len(articles)}")
print(f"Recent documents: {len(recent_articles)}")

source_names = set()

for article in recent_articles:
    source = str(article.get("source", "")).lower()
    source_names.update(tokenize(source))

trend_data = defaultdict(lambda: {
    "frequency": 0,
    "sources": set(),
    "score": 0.0,
    "article_ids": []
})

for article in recent_articles:
    title = clean_text(article.get("title", ""))
    content = clean_text(article.get("content", ""))

    words = tokenize(f"{title} {content}")

    age = article["_age_days"]
    recency_weight = math.exp(-age / 45)

    source = str(article.get("source", "Unknown"))
    article_id = article.get("id")

    for word in words:
        if word in source_names:
            continue

        trend_data[word]["frequency"] += 1
        trend_data[word]["sources"].add(source)
        trend_data[word]["score"] += recency_weight

        if article_id:
            trend_data[word]["article_ids"].append(article_id)

results = []

for word, data in trend_data.items():
    frequency = data["frequency"]
    sources = len(data["sources"])

    if frequency < MIN_FREQUENCY:
        continue

    source_weight = math.log1p(sources)
    score = data["score"] * source_weight

    results.append({
        "trend": word,
        "type": "word",
        "frequency": frequency,
        "sources": sources,
        "score": round(score, 6),
        "article_ids": data["article_ids"][:30]
    })

results.sort(key=lambda x: x["score"], reverse=True)

top_trends = results[:TOP_N]

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(top_trends, f, ensure_ascii=False, indent=2)

print("\nTop Trends:")

for trend in top_trends:
    print(
        f"{trend['trend']} | "
        f"Type: {trend['type']} | "
        f"Frequency: {trend['frequency']} | "
        f"Sources: {trend['sources']} | "
        f"Score: {trend['score']}"
    )

print(f"\nSaved to: {OUTPUT_FILE}")

