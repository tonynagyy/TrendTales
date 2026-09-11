# Trend Score = Frequency × Recency × Source Diversity

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

import nltk

nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords


stop_words = set(stopwords.words("english"))

news_stop_words = {
    "said", "says", "reported", "reportedly", "according", "added",
    "told", "announced", "stated", "confirmed", "revealed",
    "reuters", "bloomberg", "associated", "press", "news",
    "new", "nearly", "billion", "million", "trillion", "thousand",
    "percent", "percentage",
    "friday", "saturday", "sunday", "monday", "tuesday",
    "wednesday", "thursday", "january", "february", "march",
    "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "year", "years", "week",
    "weeks", "month", "months", "today", "yesterday",
    "first", "last", "next", "post", "also", "just", "still",
    "even", "back", "make", "take", "made", "been", "into",
    "over", "more", "than", "from", "with", "that", "this",
    "will", "have", "after", "about", "could", "would", "should",
    "one", "report", "reports", "two", "three", "four", "five",
    "company", "companies", "offering", "plans", "rate", "rates",
    "data", "share", "shares", "time", "well", "high", "part",
    "season", "game", "games", "team", "teams", "play", "played",
    "win", "won", "loss", "match"
}

stop_words.update(news_stop_words)


def load_documents(filename="Data preprocessing/Data/cleaned_data.json"):
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_words(text):
    text = re.sub(r"\[\+\d+\s+chars\]", "", text or "")
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return [
    word
    for word in words
    if word not in stop_words and word not in {"com", "www", "http", "https"}
    ]


def extract_phrases(words):
    phrases = []

    for i in range(len(words) - 1):
        word1 = words[i]
        word2 = words[i + 1]

        if len(word1) >= 4 and len(word2) >= 4:
            phrases.append(f"{word1} {word2}")

    return phrases


def calculate_recency(date):
    if not date:
        return 0

    try:
        article_date = datetime.fromisoformat(
            date.replace("Z", "+00:00")
        )

        now = datetime.now(article_date.tzinfo)

        hours = (now - article_date).total_seconds() / 3600

        if hours < 0:
            hours = 0

        return 1 / (1 + hours)

    except (ValueError, TypeError):
        return 0


def calculate_trend_scores(documents):
    word_frequency = Counter()
    phrase_frequency = Counter()

    word_sources = defaultdict(set)
    phrase_sources = defaultdict(set)

    word_recency = defaultdict(list)
    phrase_recency = defaultdict(list)

    for document in documents:
        content = document.get("content") or ""
        source = document.get("source") or "Unknown"
        date = document.get("date") or ""

        recency = calculate_recency(date)

        words = extract_words(content)
        phrases = extract_phrases(words)

        unique_words = set(words)
        unique_phrases = set(phrases)

        word_frequency.update(words)
        phrase_frequency.update(phrases)

        for word in unique_words:
            word_sources[word].add(source)

            if recency > 0:
                word_recency[word].append(recency)

        for phrase in unique_phrases:
            phrase_sources[phrase].add(source)

            if recency > 0:
                phrase_recency[phrase].append(recency)

    trends = []

    for word, frequency in word_frequency.items():
        recency_values = word_recency[word]

        if recency_values:
            recency = sum(recency_values) / len(recency_values)
        else:
            recency = 0

        source_diversity = len(word_sources[word])

        score = frequency * recency * source_diversity

        trends.append({
            "trend": word,
            "type": "word",
            "frequency": frequency,
            "recency": recency,
            "source_diversity": source_diversity,
            "score": score
        })

    for phrase, frequency in phrase_frequency.items():
        if frequency < 2:
            continue

        recency_values = phrase_recency[phrase]

        if not recency_values:
            continue

        recency = sum(recency_values) / len(recency_values)

        source_diversity = len(phrase_sources[phrase])

        score = frequency * recency * source_diversity

        trends.append({
            "trend": phrase,
            "type": "phrase",
            "frequency": frequency,
            "recency": recency,
            "source_diversity": source_diversity,
            "score": score
        })

    trends.sort(key=lambda x: x["score"], reverse=True)

    return trends


def save_top_trends(
    trends,
    filename="Data preprocessing/Data/top_trends.json",
    limit=10
):
    top_trends = trends[:limit]

    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(
            top_trends,
            file,
            ensure_ascii=False,
            indent=4
        )


def main():
    documents = load_documents()

    print("Documents loaded:", len(documents))

    trends = calculate_trend_scores(documents)

    print("\nTop Trends:")

    for trend in trends[:10]:
        print(
            trend["trend"],
            "| Type:", trend["type"],
            "| Frequency:", trend["frequency"],
            "| Sources:", trend["source_diversity"],
            "| Score:", round(trend["score"], 6)
        )

    save_top_trends(trends)

    print("\nTop trends saved successfully.")


if __name__ == "__main__":
    main()
