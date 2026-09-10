import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import faiss
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATA_FILE = PROJECT_ROOT / "Data preprocessing" / "Data" / "classified_data.json"
FALLBACK_DATA_FILE = PROJECT_ROOT / "Data preprocessing" / "Data" / "cleaned_data.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 100
CHUNK_OVERLAP = 20

NUM_QUERIES = 3
TOP_K_PER_QUERY = 5
FINAL_TOP_K = 3

INDEX_DIR = str(PROJECT_ROOT / "rag_index")


# DOCUMENT LOADER
def load_documents(file_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Loads documents from classified_data.json (or cleaned_data.json).
    Preserves all rich metadata including multi-label categories, confidences, and trends.
    """
    if file_path is None:
        if DEFAULT_DATA_FILE.exists():
            target_path = DEFAULT_DATA_FILE
        elif FALLBACK_DATA_FILE.exists():
            target_path = FALLBACK_DATA_FILE
        else:
            raise FileNotFoundError(f"Neither {DEFAULT_DATA_FILE} nor {FALLBACK_DATA_FILE} was found.")
    else:
        target_path = Path(file_path)

    if not target_path.exists():
        raise FileNotFoundError(f"Dataset not found: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"'{target_path}' is not valid JSON: {e}")

    if isinstance(data, dict):
        if "documents" in data:
            data = data["documents"]
        else:
            raise ValueError("JSON file does not contain a 'documents' list.")

    if not isinstance(data, list):
        raise ValueError("Expected a list of documents in JSON.")

    documents = []
    skipped = 0

    for item in data:
        if not isinstance(item, dict):
            skipped += 1
            continue

        title = str(item.get("title", "")).strip()
        content = str(item.get("content", "") or item.get("text", "")).strip()

        if not title and not content:
            skipped += 1
            continue

        categories = item.get("categories", [])
        if not categories and item.get("category"):
            categories = [item.get("category")]

        doc_id = str(item.get("id") or hashlib.md5(f"{title}|{content}".encode("utf-8")).hexdigest())

        document = {
            "id": doc_id,
            "source": str(item.get("source", "")),
            "title": title,
            "content": content,
            "date": str(item.get("date", "")),
            "url": str(item.get("url", "")),
            "category": str(item.get("category", "")),
            "categories": categories,
            "category_hint": str(item.get("category_hint", "")),
            "confidence": item.get("confidence", 0.0),
            "all_confidences": item.get("all_confidences", {}),
            "trend": item.get("trend"),
            "trend_score": item.get("trend_score", 0.0),
        }

        documents.append(document)

    if skipped:
        print(f"[warn] Skipped {skipped} malformed/empty entries in '{target_path.name}'.")

    print(f"[info] Loaded {len(documents)} documents for RAG.")
    return documents


# MULTI-QUERY EXPANSION (FREE HF API + HEURISTIC FALLBACK)
def generate_multi_queries(query: str, num_queries: int = NUM_QUERIES, category: Optional[str] = None) -> List[str]:
    """
    Produces diverse search query angles to improve semantic retrieval.
    Tries free Hugging Face API if HUGGINGFACE_TOKEN is available;
    always falls back to smart linguistic query variations.
    """
    clean_query = query.strip()
    if not clean_query:
        return []

    # Smart templates (free, 0ms, zero-cost)
    fallback = [clean_query]
    if category and category.lower() not in clean_query.lower() and category.lower() != "all":
        fallback.append(f"{clean_query} ({category} news and updates)")
    fallback.append(f"Latest news, impact and key developments about {clean_query}")
    fallback.append(f"Recent events, reports and facts regarding {clean_query}")

    # If Hugging Face token is present, try fast LLM expansion
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        try:
            # pyrefly: ignore [missing-import]
            from huggingface_hub import InferenceClient

            client = InferenceClient(api_key=hf_token)
            prompt = (
                f"Generate {num_queries - 1} alternate search queries to retrieve news about:\n"
                f"Topic: \"{clean_query}\"\n"
                f"Format your response ONLY as a JSON list of strings. Example: [\"query 1\", \"query 2\"]"
            )
            response = client.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7,
            )
            content = response.choices[0].message.content.strip()
            content = re.sub(r"^```(json)?|```$", "", content, flags=re.MULTILINE).strip()
            parsed = json.loads(content)
            if isinstance(parsed, list) and parsed:
                return [clean_query] + [str(q) for q in parsed][:num_queries - 1]
        except Exception:
            pass  # Seamlessly fall back to templates

    return fallback[:num_queries]


# MULTI-QUERY RAG ENGINE
class MultiQueryRAG:
    """
    Multi-Query RAG implementation:
    - Sentence-aware overlapping chunking
    - Hash-validated FAISS vector caching
    - Multi-angle query search with deduplication
    - Supports optional category filtering
    """

    def __init__(
        self,
        documents: List[Dict[str, Any]],
        model_name: str = EMBEDDING_MODEL,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        index_dir: str = INDEX_DIR,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self.documents = documents
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index_dir = index_dir

        print(f"[info] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

        if self._cache_exists():
            print("[info] Loading existing FAISS cache from disk...")
            self._load_cache()
        else:
            print("[info] Building new FAISS RAG index...")
            if not self.documents:
                raise ValueError("No documents provided and no valid cache found.")

            self.chunks = self._create_chunks()
            if not self.chunks:
                raise ValueError("No valid text found in documents.")

            print(f"[info] Created {len(self.chunks)} chunks.")
            embeddings = self._create_embeddings()
            self.index = self._create_faiss_index(embeddings)
            self._save_cache()

    # Cache handling
    def _cache_paths(self) -> Tuple[str, str, str]:
        return (
            os.path.join(self.index_dir, "index.faiss"),
            os.path.join(self.index_dir, "chunks.json"),
            os.path.join(self.index_dir, "cache_info.json"),
        )

    def _get_documents_hash(self) -> str:
        cache_info_path = os.path.join(self.index_dir, "cache_info.json")
        if os.path.exists(cache_info_path):
            try:
                with open(cache_info_path, "r", encoding="utf-8") as f:
                    info = json.load(f)
                    if "documents_hash" in info:
                        return info["documents_hash"]
            except Exception:
                pass
        return hashlib.md5(f"{len(self.documents)}_{self.chunk_size}".encode()).hexdigest()

    def _get_cache_info(self) -> Dict[str, Any]:
        return {
            "documents_hash": self._get_documents_hash(),
            "embedding_model": self.model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

    def _cache_exists(self) -> bool:
        index_path, chunks_path, info_path = self._cache_paths()
        if not (os.path.exists(index_path) and os.path.exists(chunks_path) and os.path.exists(info_path)):
            return False

        try:
            with open(info_path, "r", encoding="utf-8") as f:
                saved_info = json.load(f)
            current_info = self._get_cache_info()
            if saved_info != current_info:
                print("[info] Dataset or settings changed. Rebuilding index...")
                return False
            return True
        except Exception:
            return False

    def _save_cache(self):
        os.makedirs(self.index_dir, exist_ok=True)
        index_path, chunks_path, info_path = self._cache_paths()
        try:
            faiss.write_index(self.index, index_path)
            with open(chunks_path, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, ensure_ascii=False)
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(self._get_cache_info(), f, ensure_ascii=False, indent=2)
            print("[info] FAISS index successfully cached.")
        except Exception as e:
            print(f"[warning] Could not save cache: {e}")

    def _load_cache(self):
        index_path, chunks_path, _ = self._cache_paths()
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        print(f"[info] Loaded {len(self.chunks)} chunks from cache.")

    # Chunking
    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _create_chunks(self) -> List[Dict[str, Any]]:
        chunks = []
        for document in self.documents:
            title = document.get("title", "")
            content = document.get("content", "")
            full_text = f"{title}\n{content}".strip()
            if not full_text:
                continue

            sentences = self._split_sentences(full_text)
            if not sentences:
                continue

            current_words = []
            chunk_number = 0

            def make_chunk(words, number):
                return {
                    "chunk_id": f"{document['id']}_{number}",
                    "id": document["id"],
                    "source": document.get("source", ""),
                    "title": document.get("title", ""),
                    "content": document.get("content", ""),
                    "date": document.get("date", ""),
                    "url": document.get("url", ""),
                    "category": document.get("category", ""),
                    "categories": document.get("categories", []),
                    "category_hint": document.get("category_hint", ""),
                    "trend_score": document.get("trend_score", 0.0),
                    "text": " ".join(words),
                }

            for sentence in sentences:
                sentence_words = sentence.split()
                if len(sentence_words) >= self.chunk_size:
                    if current_words:
                        chunks.append(make_chunk(current_words, chunk_number))
                        chunk_number += 1
                        current_words = []
                    chunks.append(make_chunk(sentence_words, chunk_number))
                    chunk_number += 1
                    continue

                if len(current_words) + len(sentence_words) > self.chunk_size and current_words:
                    chunks.append(make_chunk(current_words, chunk_number))
                    chunk_number += 1
                    overlap_words = current_words[-self.chunk_overlap:] if self.chunk_overlap else []
                    current_words = overlap_words + sentence_words
                else:
                    current_words.extend(sentence_words)

            if current_words:
                chunks.append(make_chunk(current_words, chunk_number))

        return chunks

    def _create_embeddings(self) -> np.ndarray:
        texts = [chunk["text"] for chunk in self.chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def _create_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        return index

    def search_query(self, query: str, top_k: int = TOP_K_PER_QUERY) -> List[Dict[str, Any]]:
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        top_k = min(top_k, self.index.ntotal)
        if top_k <= 0:
            return []

        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for score, index_id in zip(scores[0], indices[0]):
            if index_id == -1:
                continue
            chunk = dict(self.chunks[index_id])
            chunk["similarity"] = float(score)
            chunk["matched_query"] = query
            results.append(chunk)

        return results

    def retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        final_top_k: int = FINAL_TOP_K,
    ) -> List[Dict[str, Any]]:
        """
        Multi-Query Retrieval:
        1. Generates 3 query variations
        2. Retrieves candidate chunks per query
        3. Fuses & deduplicates by document ID
        4. Applies optional category filtering / soft boosting
        5. Returns top K articles (default: 3)
        """
        if not query or not query.strip():
            return []

        queries = generate_multi_queries(query, num_queries=NUM_QUERIES, category=category)

        all_results = []
        # Search more candidates if we want to filter by category
        search_top_k = TOP_K_PER_QUERY * 3 if category and category.lower() != "all" else TOP_K_PER_QUERY
        for q in queries:
            all_results.extend(self.search_query(q, top_k=search_top_k))

        # Deduplicate by document ID (keep highest similarity)
        unique_results: Dict[str, Dict[str, Any]] = {}
        for result in all_results:
            doc_id = result["id"]
            sim = result["similarity"]

            # Category matching boost/filtering
            if category and category.lower() != "all":
                doc_categories = [c.lower() for c in result.get("categories", [])]
                if category.lower() in doc_categories or category.lower() == result.get("category", "").lower():
                    sim += 0.15  # Soft boost for matching category

            if doc_id not in unique_results or sim > unique_results[doc_id]["similarity"]:
                result_copy = dict(result)
                result_copy["similarity"] = sim
                unique_results[doc_id] = result_copy

        ranked_results = sorted(
            unique_results.values(),
            key=lambda x: x["similarity"],
            reverse=True,
        )

        return ranked_results[:final_top_k]

    def build_context(
        self,
        query: str,
        category: Optional[str] = None,
        final_top_k: int = FINAL_TOP_K,
    ) -> Tuple[List[Dict[str, Any]], str]:
        results = self.retrieve(query, category=category, final_top_k=final_top_k)

        context_parts = []
        for i, result in enumerate(results, start=1):
            categories_str = ", ".join(result.get("categories", [])) or result.get("category", "General")
            desc = result.get("content") or result.get("text", "")
            result["content"] = desc
            context_parts.append(
                f"--- Event {i} ---\n"
                f"Title: {result.get('title', 'Untitled')}\n"
                f"Source: {result.get('source', '')}\n"
                f"Date: {result.get('date', '')}\n"
                f"Category: {categories_str}\n"
                f"Similarity: {result.get('similarity', 0.0):.4f}\n"
                f"Description: {desc}\n"
            )

        context = "\n".join(context_parts)
        return results, context


if __name__ == "__main__":
    docs = load_documents()
    rag = MultiQueryRAG(docs)
    test_query = "Artificial Intelligence"
    res, ctx = rag.build_context(test_query, final_top_k=3)
    print(f"\nRetrieved {len(res)} events for '{test_query}':")
    for r in res:
        print(f"- {r['title']} (sim: {r['similarity']:.4f})")
