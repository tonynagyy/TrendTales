import os
import re
import json
import hashlib

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import faiss
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer


# =========================================================
# SETTINGS
# =========================================================

DATA_FILE = "c:\\Users\\Saed\\Documents\\rag_project\\cleaned_data.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 100
CHUNK_OVERLAP = 20

NUM_QUERIES = 3
TOP_K_PER_QUERY = 5
FINAL_TOP_K = 5

INDEX_DIR = "rag_index"

ANTHROPIC_MODEL = "claude-sonnet-4-6"


# =========================================================
# LOAD DATA
# =========================================================

def load_documents(file_path):

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"'{file_path}' is not valid JSON: {e}")

    # Support both:
    # [...]
    # and
    # {"documents": [...]}
    if isinstance(data, dict):
        if "documents" in data:
            data = data["documents"]
        else:
            raise ValueError("JSON file does not contain 'documents'.")

    if not isinstance(data, list):
        raise ValueError("Expected a list of documents (or a 'documents' list).")

    documents = []
    skipped = 0

    for item in data:

        # Guard against malformed entries instead of crashing on item.get(...)
        if not isinstance(item, dict):
            skipped += 1
            continue

        title = str(item.get("title", "")).strip()
        content = str(item.get("content", "")).strip()

        if not title and not content:
            skipped += 1
            continue

        document = {
            "id": item.get("id", ""),
            "source": item.get("source", ""),
            "title": title,
            "content": content,
            "date": item.get("date", ""),
            "url": item.get("url", ""),
            "category_hint": item.get("category_hint", ""),
        }

        documents.append(document)

    if skipped:
        print(f"[warn] Skipped {skipped} malformed/empty entries in '{file_path}'.")

    print(f"[info] Loaded {len(documents)} documents.")

    return documents


# =========================================================
# LLM QUERY EXPANSION
# =========================================================

def llm_generate_queries(query, num_queries=NUM_QUERIES):
    """
    Ask an LLM to produce alternate phrasings / angles of the user's query.
    Falls back to simple templated queries if the API isn't available or fails.
    """

    fallback = [
        query,
        f"Latest news and developments about {query}",
        f"Important facts, events, impact and reports about {query}",
    ][:num_queries]

    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        return fallback

    try:
        # pyrefly: ignore [missing-import]
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        prompt = (
            f"Generate {num_queries - 1} alternate search queries that would help "
            f"retrieve relevant documents for the following topic. "
            f"Each alternate query should approach the topic from a different angle "
            f"(e.g. a different sub-aspect, a more specific or more general phrasing).\n\n"
            f"Original topic: \"{query}\"\n\n"
            f"Respond ONLY with a JSON array of strings, nothing else. "
            f"Example: [\"query one\", \"query two\"]"
        )

        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )

        text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()

        text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()

        alt_queries = json.loads(text)

        if not isinstance(alt_queries, list) or not alt_queries:
            return fallback

        queries = [query] + [str(q) for q in alt_queries]
        return queries[:num_queries]

    except Exception as e:
        print(f"[warn] LLM query generation failed ({e}); using fallback templates.")
        return fallback


# ===============
# MULTI QUERY RAG

class MultiQueryRAG:

    def __init__(
        self,
        documents,
        model_name=EMBEDDING_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        index_dir=INDEX_DIR,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self.documents = documents
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index_dir = index_dir

        print(f"[info] Loading embedding model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model '{model_name}': {e}")

        # Check cache (hash-validated: rebuilds automatically if data/settings changed)
        if self._cache_exists():
            print("[info] Loading existing FAISS cache...")
            self._load_cache()
        else:
            print("[info] Building new RAG index...")

            if not self.documents:
                raise ValueError("No documents provided and no valid cache found.")

            self.chunks = self._create_chunks()
            if not self.chunks:
                raise ValueError("No valid text was found in the provided documents.")

            print(f"[info] Created {len(self.chunks)} chunks.")

            embeddings = self._create_embeddings()
            self.index = self._create_faiss_index(embeddings)
            self._save_cache()

    # ===================
    # CACHE (hash-validated: rebuilds if data or settings change)

    def _cache_paths(self):
        return (
            os.path.join(self.index_dir, "index.faiss"),
            os.path.join(self.index_dir, "chunks.json"),
            os.path.join(self.index_dir, "cache_info.json"),
        )

    def _get_documents_hash(self):
        data = json.dumps(self.documents, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def _get_cache_info(self):
        return {
            "documents_hash": self._get_documents_hash(),
            "embedding_model": self.model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

    def _cache_exists(self):
        index_path, chunks_path, info_path = self._cache_paths()

        if not (os.path.exists(index_path) and os.path.exists(chunks_path) and os.path.exists(info_path)):
            return False

        try:
            with open(info_path, "r", encoding="utf-8") as f:
                saved_info = json.load(f)

            current_info = self._get_cache_info()

            if saved_info != current_info:
                print("[info] Dataset or settings changed.")
                print("[info] Rebuilding FAISS index...")
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
                json.dump(self.chunks, f, ensure_ascii=False, indent=2)

            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(self._get_cache_info(), f, ensure_ascii=False, indent=2)

            print("[info] FAISS index saved.")

        except Exception as e:
            print(f"[warning] Could not save cache: {e}")

    def _load_cache(self):
        index_path, chunks_path, info_path = self._cache_paths()

        try:
            self.index = faiss.read_index(index_path)

            with open(chunks_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)

            print(f"[info] Loaded {len(self.chunks)} chunks from cache.")

        except Exception as e:
            raise RuntimeError(f"Failed to load cache: {e}")

    # =====================================================
    # CHUNKING (sentence-aware)
    # =====================================================

    def _split_sentences(self, text):
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _create_chunks(self):
        chunks = []

        for document in self.documents:
            title = document["title"]
            content = document["content"]

            full_text = (title + "\n" + content).strip()
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
                    "source": document["source"],
                    "title": document["title"],
                    "date": document["date"],
                    "url": document["url"],
                    "category_hint": document["category_hint"],
                    "text": " ".join(words),
                }

            for sentence in sentences:
                sentence_words = sentence.split()

                # A single sentence longer than chunk_size stands on its own
                # instead of being cut mid-sentence.
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

    # EMBEDDINGS
    # =====================================================

    def _create_embeddings(self):
        texts = [chunk["text"] for chunk in self.chunks]

        if not texts:
            raise ValueError("No valid text was found.")

        print("[info] Creating embeddings...")

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception as e:
            raise RuntimeError(f"Embedding generation failed: {e}")

        return embeddings.astype(np.float32)

    # =====================================================
    # FAISS
    # =====================================================

    def _create_faiss_index(self, embeddings):
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        print(f"[info] FAISS index contains {index.ntotal} vectors.")
        return index

    # =====================================================
    # MULTI QUERY (LLM-backed, with fallback)
    # =====================================================

    def generate_queries(self, query):
        return llm_generate_queries(query, NUM_QUERIES)

    # =====================================================
    # SEARCH ONE QUERY
    # =====================================================

    def search_query(self, query, top_k=TOP_K_PER_QUERY):
        try:
            query_embedding = self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
        except Exception as e:
            print(f"[warn] Failed to embed query '{query}': {e}")
            return []

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

    # =====================================================
    # MULTI QUERY RETRIEVAL
    # =====================================================

    def retrieve(self, query, final_top_k=FINAL_TOP_K):
        if not query or not query.strip():
            return []

        queries = self.generate_queries(query)

        print("\nQueries used:")
        for q in queries:
            print(" -", q)

        all_results = []
        for q in queries:
            all_results.extend(self.search_query(q))

        # Remove duplicates, keep the highest similarity per chunk
        unique_results = {}
        for result in all_results:
            key = (result["id"], result["text"])
            if key not in unique_results or result["similarity"] > unique_results[key]["similarity"]:
                unique_results[key] = result

        ranked_results = sorted(
            unique_results.values(),
            key=lambda x: x["similarity"],
            reverse=True,
        )

        return ranked_results[:final_top_k]

    # =====================================================
    # BUILD CONTEXT
    # =====================================================

    def build_context(self, query, final_top_k=FINAL_TOP_K):
        results = self.retrieve(query, final_top_k)

        context_parts = []

        for i, result in enumerate(results, start=1):
            context_parts.append(
                f"""
--- Document {i} ---

Title:
{result["title"]}

Source:
{result["source"]}

Date:
{result["date"]}

Category:
{result["category_hint"]}

Similarity:
{result["similarity"]:.4f}

Content:
{result["text"]}
"""
            )

        context = "\n".join(context_parts)

        return results, context


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TrendTales AI - Multi Query RAG")
    print("=" * 60)

    documents = load_documents(DATA_FILE)

    rag = MultiQueryRAG(documents)

    query = input("\nEnter your topic/query: ").strip()

    if not query:
        print("[error] Empty query — nothing to search for.")
    else:
        results, context = rag.build_context(query)

        print("\n")
        print("=" * 60)
        print("TOP RESULTS")
        print("=" * 60)

        if not results:
            print("No results found.")
        else:
            for i, result in enumerate(results, start=1):
                print(f"\n{i}. {result['title']}")
                print(f"Source: {result['source']}")
                print(f"Similarity: {result['similarity']:.4f}")
                print(f"Matched Query: {result['matched_query']}")

        print("\n")
        print("=" * 60)
        print("CONTEXT FOR LLM")
        print("=" * 60)
        print(context)