import os
import re
import json
import hashlib

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


CLEANED_DATA_FILE = "Data preprocessing/Data/cleaned_data.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 100
CHUNK_OVERLAP = 20

NUM_QUERIES = 3

INDEX_DIR = "rag_index"

FINAL_TOP_K = None


def load_documents(filename=CLEANED_DATA_FILE):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        documents = data.get("documents", [])
    elif isinstance(data, list):
        documents = data
    else:
        raise ValueError("Invalid JSON format.")

    cleaned_documents = []

    for i, doc in enumerate(documents):

        if not isinstance(doc, dict):
            continue

        content = str(
            doc.get("content")
            or doc.get("text")
            or doc.get("title")
            or ""
        ).strip()

        title = str(
            doc.get("title")
            or "Untitled Article"
        ).strip()

        if not content:
            continue

        article_id = str(
            doc.get("id")
            or hashlib.md5(
                f"{title}|{content}".encode("utf-8")
            ).hexdigest()
        )

        cleaned_documents.append(
            {
                "id": article_id,
                "source": str(doc.get("source") or ""),
                "title": title,
                "content": content,
                "date": str(doc.get("date") or ""),
                "url": str(doc.get("url") or ""),
                "category_hint": str(
                    doc.get("category_hint") or ""
                ),
                "trend": doc.get("trend"),
                "trend_type": doc.get("trend_type"),
                "trend_frequency": doc.get("trend_frequency"),
                "trend_sources": doc.get("trend_sources"),
                "trend_score": doc.get("trend_score"),
            }
        )

    return cleaned_documents


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class MultiQueryRAG:

    def __init__(
        self,
        documents,
        embedding_model=EMBEDDING_MODEL,
        index_dir=INDEX_DIR
    ):

        self.documents = documents
        self.embedding_model_name = embedding_model
        self.index_dir = index_dir

        self.model = SentenceTransformer(
            self.embedding_model_name
        )

        self.chunks = []
        self.index = None

        self._prepare_chunks()
        self._build_index()

    def _prepare_chunks(self):

        self.chunks = []

        for document in self.documents:

            content = clean_text(
                document["content"]
            )

            words = content.split()

            if not words:
                continue

            start = 0

            while start < len(words):

                end = min(
                    start + CHUNK_SIZE,
                    len(words)
                )

                chunk_words = words[start:end]

                if not chunk_words:
                    break

                chunk_text = " ".join(
                    chunk_words
                )

                chunk = {
                    "chunk_id": (
                        f"{document['id']}_{start}"
                    ),
                    "id": document["id"],
                    "source": document["source"],
                    "title": document["title"],
                    "content": document["content"],
                    "date": document["date"],
                    "url": document["url"],
                    "category_hint": document[
                        "category_hint"
                    ],
                    "trend": document.get("trend"),
                    "trend_type": document.get(
                        "trend_type"
                    ),
                    "trend_frequency": document.get(
                        "trend_frequency"
                    ),
                    "trend_sources": document.get(
                        "trend_sources"
                    ),
                    "trend_score": document.get(
                        "trend_score"
                    ),
                    "text": chunk_text,
                }

                self.chunks.append(chunk)

                if end >= len(words):
                    break

                start = end - CHUNK_OVERLAP

                if start < 0:
                    start = 0

    def _build_index(self):

        if not self.chunks:
            raise RuntimeError(
                "No chunks were created from the documents."
            )

        texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        embeddings = embeddings.astype(
            np.float32
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(embeddings)

    def generate_queries(
        self,
        query
    ):

        query = query.strip()

        queries = [
            query,
            f"{query} latest news",
            f"{query} recent developments"
        ]

        return queries[:NUM_QUERIES]

    def retrieve(
        self,
        query,
        top_k=None
    ):

        if not self.chunks:
            return []

        queries = self.generate_queries(
            query
        )

        all_results = []

        search_k = len(self.chunks)

        for generated_query in queries:

            query_embedding = self.model.encode(
                [generated_query],
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype(np.float32)

            scores, indices = self.index.search(
                query_embedding,
                search_k
            )

            for score, idx in zip(
                scores[0],
                indices[0]
            ):

                if idx < 0:
                    continue

                chunk = dict(
                    self.chunks[idx]
                )

                chunk["similarity"] = float(
                    score
                )

                chunk["matched_query"] = (
                    generated_query
                )

                all_results.append(chunk)

        article_results = {}

        for result in all_results:

            article_id = result["id"]

            score = result["similarity"]

            if (
                article_id not in article_results
                or score
                > article_results[article_id][
                    "similarity"
                ]
            ):

                article_results[
                    article_id
                ] = result

        results = list(
            article_results.values()
        )

        results.sort(
            key=lambda x: x["similarity"],
            reverse=True
        )

        if top_k is not None:
            results = results[:top_k]

        return results

    def build_context(
        self,
        query,
        final_top_k=None
    ):

        results = self.retrieve(
            query,
            top_k=final_top_k
        )

        context_parts = []

        for i, result in enumerate(
            results,
            start=1
        ):

            context_parts.append(
                f"""
Article {i}

Title:
{result["title"]}

Date:
{result["date"]}

Source:
{result["source"]}

Content:
{result["content"]}

URL:
{result["url"]}
""".strip()
            )

        context = "\n\n".join(
            context_parts
        )

        return results, context


if __name__ == "__main__":

    print("=" * 60)
    print("TrendTales RAG")
    print("=" * 60)

    documents = load_documents(
        CLEANED_DATA_FILE
    )

    print(
        f"Loaded {len(documents)} articles."
    )

    rag = MultiQueryRAG(
        documents
    )

    query = input(
        "\nEnter your topic: "
    ).strip()

    results, context = rag.build_context(
        query,
        final_top_k=None
    )

    print(
        f"\nFound {len(results)} unique articles."
    )

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\n{i}. {result['title']}"
        )

        print(
            f"   Date: {result['date']}"
        )

        print(
            f"   Similarity: "
            f"{result['similarity']:.4f}"
        )