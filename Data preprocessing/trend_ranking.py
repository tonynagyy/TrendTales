# Trend Score = Frequency × Recency × Source Diversity

import json
import re
# pyrefly: ignore [missing-import]
import nltk
from pathlib import Path
# pyrefly: ignore [missing-import]
from nltk.corpus import stopwords
from collections import Counter
from datetime import datetime

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

news_stop_words = {
    # Reporting verbs
    "said", "says", "reported", "reportedly", "according", "added",
    "told", "announced", "stated", "confirmed", "revealed",
    # News sources / generic names
    "reuters", "bloomberg", "associated", "press", "news",
    # Vague quantities
    "new", "nearly", "billion", "million", "trillion", "thousand",
    "percent", "percentage",
    # Days / months / time
    "friday", "saturday", "sunday", "monday", "tuesday",
    "wednesday", "thursday", "january", "february", "march",
    "april", "june", "july", "august", "september", "october",
    "november", "december", "year", "years", "week", "weeks",
    "month", "months", "today", "yesterday",
    # Generic filler words that appeared in rankings
    "first", "last", "next", "post", "also", "just", "still",
    "even", "back", "make", "take", "made", "been", "into",
    "over", "more", "than", "from", "with", "that", "this",
    "will", "have", "after", "about", "could", "would", "should",
    "one", "report", "reports", "two", "three", "four", "five",
    # Business filler
    "company", "companies", "offering", "plans", "rate", "rates",
    "data", "share", "shares", "time", "well", "high", "part",
    # Sports filler
    "season", "game", "games", "team", "teams", "play", "played",
    "win", "won", "loss", "match",
}

stop_words.update(news_stop_words)


def load_documents(filename="Data preprocessing/Data/cleaned_data.json"):
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)

def extract_words(text):
    text = re.sub(r"\[\+\d+\s+chars\]", "", text)
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

    return [word for word in words if word not in stop_words]

def extract_phrases(text):
    words = extract_words(text)
    phrases = []

    for i in range(len(words) - 1):
        word1 = words[i]
        word2 = words[i + 1]

        if len(word1) >= 4 and len(word2) >= 4:
            phrases.append(f"{word1} {word2}")

    return phrases

def calculate_frequency(documents):
    word_counter = Counter()
    phrase_counter = Counter()

    for document in documents:
        text = document["content"]

        word_counter.update(extract_words(text))
        phrase_counter.update(extract_phrases(text))

    return word_counter, phrase_counter

def calculate_recency(date):
    article_date = datetime.fromisoformat(date.replace("Z", "+00:00"))
    now = datetime.now(article_date.tzinfo)

    hours = (now - article_date).total_seconds() / 3600

    return 1 / (1 + hours)

def calculate_source_diversity(documents):
    word_sources = {}
    phrase_sources = {}

    for document in documents:
        text = document["content"]
        source = document["source"]

        words = set(extract_words(text))
        phrases = set(extract_phrases(text))

        for word in words:
            if word not in word_sources:
                word_sources[word] = set()
            word_sources[word].add(source)

        for phrase in phrases:
            if phrase not in phrase_sources:
                phrase_sources[phrase] = set()
            phrase_sources[phrase].add(source)

    return word_sources, phrase_sources

def calculate_trend_scores(documents):
    word_counter, phrase_counter = calculate_frequency(documents)
    word_sources, phrase_sources = calculate_source_diversity(documents)

    trends = []

    for word, frequency in word_counter.items():
        source_diversity = len(word_sources[word])

        recency_scores = []

        for document in documents:
            if word in extract_words(document["content"]):
                recency_scores.append(calculate_recency(document["date"]))

        recency = sum(recency_scores) / len(recency_scores)

        score = frequency * recency * source_diversity

        trends.append({
            "trend": word,
            "type": "word",
            "frequency": frequency,
            "recency": recency,
            "source_diversity": source_diversity,
            "score": score
        })

    for phrase, frequency in phrase_counter.items():
        if frequency < 2:
            continue

        source_diversity = len(phrase_sources[phrase])

        recency_scores = []

        for document in documents:
            if phrase in " ".join(extract_words(document["content"])):
                recency_scores.append(calculate_recency(document["date"]))

        if not recency_scores:
            continue

        recency = sum(recency_scores) / len(recency_scores)

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

def save_top_trends(trends, filename="Data preprocessing/Data/top_trends.json", limit=10):
    top_trends = trends[:limit]

    Path(filename).parent.mkdir(parents=True, exist_ok=True)  # Fix: auto-create Data/ dir
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(top_trends, file, ensure_ascii=False, indent=4)

def main():
    documents = load_documents()

    trends = calculate_trend_scores(documents)

    print("Top Trends:")

    for trend in trends[:10]:
        print(
            trend["trend"],
            "| Type:", trend["type"],
            "| Frequency:", trend["frequency"],
            "| Sources:", trend["source_diversity"],
            "| Score:", round(trend["score"], 4)
        )

    save_top_trends(trends)

    print("\nTop trends saved successfully.")



if __name__ == "__main__":
    main()