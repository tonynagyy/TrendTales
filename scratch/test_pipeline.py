import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.multi_query_rag import load_documents, MultiQueryRAG
from story_generation.story_generator import generate_story_from_events
from bert_classifier.classify import classify_article

print("=== 1. CLASSIFY QUERY ===")
query = "artificial intelligence and school education"
cls = classify_article(query)
print("Categories:", cls["categories"])
print("Primary:", cls["primary_category"])

print("\n=== 2. MULTI-QUERY RAG RETRIEVAL ===")
docs = load_documents()
rag = MultiQueryRAG(docs)
events, context = rag.build_context(query, category=cls["primary_category"], final_top_k=3)
print(f"Retrieved {len(events)} events (Top 3):")
for i, ev in enumerate(events, 1):
    print(f"  {i}. {ev['title']} [{ev.get('source', '')}] (Sim: {ev['similarity']:.4f})")

print("\n=== 3. STORY GENERATION ===")
story = generate_story_from_events(events)
print("Story Preview (first 300 chars):")
print(story[:300] + "...")
print("\nPIPELINE TEST PASSED SUCCESSFULLY!")
