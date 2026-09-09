import json
import re
import html
import hashlib
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from datetime import datetime

from bs4 import BeautifulSoup


INPUT_FILE = "Data preprocessing/Data/final_news.json"
OUTPUT_FILE = "Data preprocessing/Data/cleaned_data.json"


HTML_BLOCK_PATTERN = re.compile(
    r"<(script|style|noscript|iframe|svg|template).*?>.*?</\1>",
    re.IGNORECASE | re.DOTALL
)

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

URL_PATTERN = re.compile(
    r"(https?://\S+|www\.\S+)",
    re.IGNORECASE
)

CHAR_PATTERN = re.compile(
    r"\[\s*\+\s*\d+\s*chars?\s*\]",
    re.IGNORECASE
)

WHITESPACE_PATTERN = re.compile(r"\s+")

ARTIFACT_PATTERN = re.compile(
    r"\b("
    r"href|target|blank|nbsp|font|color|style|class|"
    r"rss|xml|xmlns|doctype|"
    r")\b",
    re.IGNORECASE
)

TRACKING_PARAMETERS = {
    "oc",
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_name",
    "gclid",
    "fbclid",
    "mc_cid",
    "mc_eid"
}


def load_documents(filename=INPUT_FILE):
    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        if "articles" in data:
            data = data["articles"]
        elif "documents" in data:
            data = data["documents"]
        else:
            data = [data]

    if not isinstance(data, list):
        raise ValueError("Input JSON must contain a list of documents.")

    return data


def clean_html(text):
    if not text:
        return ""

    text = str(text)
    text = html.unescape(text)

    text = HTML_BLOCK_PATTERN.sub(" ", text)

    if re.search(r"<[a-zA-Z][^>]*>", text):
        soup = BeautifulSoup(text, "html.parser")

        for element in soup(
            ["script", "style", "noscript", "iframe", "svg", "template"]
        ):
            element.decompose()

        text = soup.get_text(" ", strip=True)
    else:
        text = HTML_TAG_PATTERN.sub(" ", text)

    text = html.unescape(text)

    return text


def clean_text(text, remove_urls=True):
    text = clean_html(text)

    text = CHAR_PATTERN.sub(" ", text)

    if remove_urls:
        text = URL_PATTERN.sub(" ", text)

    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", " ", text)

    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    text = WHITESPACE_PATTERN.sub(" ", text)

    return text.strip(" -|•·")


def clean_url(url):
    if not url:
        return ""

    url = str(url).strip()

    try:
        parsed = urlsplit(url)

        if not parsed.scheme or not parsed.netloc:
            return url

        query = [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if key.lower() not in TRACKING_PARAMETERS
            and not key.lower().startswith("utm_")
        ]

        cleaned_query = urlencode(query)

        return urlunsplit(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                parsed.path,
                cleaned_query,
                ""
            )
        )

    except Exception:
        return url


def normalize_for_comparison(text):
    text = clean_text(text).lower()

    text = re.sub(r"[^\w\s]", " ", text)

    text = WHITESPACE_PATTERN.sub(" ", text)

    return text.strip()


def remove_duplicate_title_from_content(title, content):
    if not title or not content:
        return content

    normalized_title = normalize_for_comparison(title)
    normalized_content = normalize_for_comparison(content)

    if not normalized_title or not normalized_content:
        return content

    if normalized_content == normalized_title:
        return ""

    title_lower = title.lower().strip()
    content_lower = content.lower().strip()

    if content_lower.startswith(title_lower):
        remaining = content[len(title):].strip(" .-|:;,")
        return remaining.strip()

    title_words = normalized_title.split()
    content_words = normalized_content.split()

    if (
        len(title_words) >= 5
        and len(content_words) > len(title_words)
        and content_words[:len(title_words)] == title_words
    ):
        original_words = content.split()
        remaining = original_words[len(title_words):]
        return " ".join(remaining).strip(" .-|:;,")

    return content


def remove_duplicate_content(content):
    if not content:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", content)

    cleaned_sentences = []
    seen = set()

    for sentence in sentences:
        normalized = normalize_for_comparison(sentence)

        if not normalized:
            continue

        if normalized in seen:
            continue

        seen.add(normalized)
        cleaned_sentences.append(sentence.strip())

    return " ".join(cleaned_sentences).strip()


def remove_source_suffix(content, source):
    if not content or not source:
        return content

    content_normalized = normalize_for_comparison(content)
    source_normalized = normalize_for_comparison(source)

    if not source_normalized:
        return content

    if content_normalized == source_normalized:
        return ""

    source_pattern = re.escape(source.strip())

    content = re.sub(
        rf"\s*[-|•·:]\s*{source_pattern}\s*$",
        "",
        content,
        flags=re.IGNORECASE
    )

    content = re.sub(
        rf"\s+{source_pattern}\s*$",
        "",
        content,
        flags=re.IGNORECASE
    )

    return content.strip(" .-|•·:;,")


def remove_rss_artifacts(text):
    if not text:
        return ""

    text = ARTIFACT_PATTERN.sub(" ", text)

    text = re.sub(
        r"\b(html|body|head|meta|link|script|style)\b",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_content(title, content, source):
    content = clean_text(content)
    content = remove_rss_artifacts(content)

    if not content:
        return ""

    content = remove_duplicate_title_from_content(title, content)

    content = remove_source_suffix(content, source)

    content = remove_duplicate_content(content)

    content = clean_text(content)

    return content


def parse_date(value):
    if not value:
        return None

    value = str(value).strip()

    if not value:
        return None

    formats = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%d %H:%M:%S",
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S GMT"
    ]

    for date_format in formats:
        try:
            parsed = datetime.strptime(value, date_format)

            if parsed.tzinfo:
                parsed = parsed.replace(tzinfo=None)

            return parsed.isoformat()
        except ValueError:
            continue

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))

        if parsed.tzinfo:
            parsed = parsed.replace(tzinfo=None)

        return parsed.isoformat()
    except ValueError:
        return None


def generate_id(document):
    base = (
        str(document.get("url", "")) +
        "|" +
        str(document.get("title", "")) +
        "|" +
        str(document.get("source", ""))
    )

    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


def preprocess_documents(documents):
    cleaned_documents = []

    seen_urls = set()
    seen_title_source = set()

    stats = {
        "html_cleaned": 0,
        "urls_cleaned": 0,
        "invalid_records": 0,
        "duplicate_urls": 0,
        "duplicate_title_source": 0,
        "empty_content_fallback": 0,
        "invalid_dates": 0
    }

    for document in documents:
        if not isinstance(document, dict):
            stats["invalid_records"] += 1
            continue

        title = clean_text(document.get("title"))
        source = clean_text(document.get("source"))
        content = clean_content(
            title,
            document.get("content"),
            source
        )

        original_content = str(document.get("content") or "")

        if original_content != content:
            if (
                "<" in original_content
                or ">" in original_content
                or "&nbsp;" in original_content
                or "href" in original_content.lower()
            ):
                stats["html_cleaned"] += 1

        original_url = document.get("url", "")
        cleaned_url = clean_url(original_url)

        if original_url != cleaned_url:
            stats["urls_cleaned"] += 1

        date = parse_date(document.get("date"))

        if document.get("date") and date is None:
            stats["invalid_dates"] += 1

        if not title:
            stats["invalid_records"] += 1
            continue

        if not content:
            content = title
            stats["empty_content_fallback"] += 1

        url_key = cleaned_url.lower().strip()

        if url_key:
            if url_key in seen_urls:
                stats["duplicate_urls"] += 1
                continue

            seen_urls.add(url_key)

        title_source_key = (
            normalize_for_comparison(title),
            normalize_for_comparison(source)
        )

        if title_source_key in seen_title_source:
            stats["duplicate_title_source"] += 1
            continue

        seen_title_source.add(title_source_key)

        cleaned_document = dict(document)

        cleaned_document["id"] = (
            str(document.get("id")).strip()
            if document.get("id")
            else generate_id(
                {
                    "url": cleaned_url,
                    "title": title,
                    "source": source
                }
            )
        )

        cleaned_document["title"] = title
        cleaned_document["content"] = content
        cleaned_document["source"] = source
        cleaned_document["url"] = cleaned_url
        cleaned_document["date"] = date

        cleaned_documents.append(cleaned_document)

    cleaned_documents.sort(
        key=lambda document: document.get("date") or "",
        reverse=True
    )

    return cleaned_documents, stats


def save_documents(documents, filename=OUTPUT_FILE):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(
            documents,
            file,
            ensure_ascii=False,
            indent=2
        )


def main():
    documents = load_documents()

    cleaned_documents, stats = preprocess_documents(documents)

    save_documents(cleaned_documents)

    print("Original documents:", len(documents))
    print("Final cleaned documents:", len(cleaned_documents))
    print("HTML/RSS cleaned:", stats["html_cleaned"])
    print("URLs cleaned:", stats["urls_cleaned"])
    print("Duplicate URLs removed:", stats["duplicate_urls"])
    print("Duplicate title/source records removed:", stats["duplicate_title_source"])
    print("Invalid records removed:", stats["invalid_records"])
    print("Empty content replaced with title:", stats["empty_content_fallback"])
    print("Invalid dates:", stats["invalid_dates"])
    print("Output:", OUTPUT_FILE)
    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()
