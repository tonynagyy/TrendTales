"""
TrendTales AI — CLI Pipeline
Usage: python main.py
Flow: User Query → BERT Classification → Multi-Query RAG → Story Generation → TTS
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bert_classifier.classify import classify_article
from rag.multi_query_rag import load_documents, MultiQueryRAG
from story_generation.story_generator import generate_story_from_events
from story_generation.tts import text_to_speech


def run_pipeline(query: str, top_k: int = 3, category: str = None):

    print("=" * 60)
    print("TrendTales AI — BERT + RAG + Story + TTS")
    print("=" * 60)

    # ── Step 1: BERT Multi-Label Classification ──────────────────────────────
    print("\n[1/4] BERT Classification...")
    cls = classify_article(query.strip())
    categories = cls["categories"]
    primary    = cls["primary_category"]
    print(f"  Detected: {categories}  (primary: {primary})")

    # ── Step 2: Multi-Query RAG Retrieval ────────────────────────────────────
    print(f"\n[2/4] RAG Retrieval  (top {top_k} events)...")
    docs = load_documents()
    rag  = MultiQueryRAG(docs)

    rag_category = category or primary
    events, context = rag.build_context(
        query,
        category=rag_category,
        final_top_k=top_k,
    )

    if not events:
        print("  [!] No relevant events found. Try a different query.")
        return

    print(f"  Retrieved {len(events)} events:")
    for i, ev in enumerate(events, 1):
        print(f"    {i}. {ev['title']}  [{ev.get('source', '')}]")

    # ── Step 3: Story Generation ─────────────────────────────────────────────
    print("\n[3/4] Story Generation...")
    story = generate_story_from_events(events)
    print("\n--- GENERATED STORY ---")
    print(story)
    print("--- END OF STORY ---")

    # ── Step 4: Text-to-Speech ────────────────────────────────────────────────
    print("\n[4/4] Text-to-Speech...")
    output_dir  = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    audio_path  = output_dir / "generated_story.mp3"

    text_to_speech(text=story, output_path=str(audio_path))
    print(f"  Audio saved to: {audio_path}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE ✓")
    print("=" * 60)

    return {
        "categories": categories,
        "events":     events,
        "story":      story,
        "audio_path": str(audio_path),
    }


if __name__ == "__main__":
    query = input("\nEnter your topic: ").strip()
    if not query:
        print("[error] Topic cannot be empty.")
        sys.exit(1)

    run_pipeline(query)