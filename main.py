import sys
from pathlib import Path
from tts import text_to_speech

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bert_classifier.classify import classify_article

from rag import (
    load_top_trend_documents,
    MultiQueryRAG,
    TOP_TRENDS_FILE,
    CLEANED_DATA_FILE,
    FINAL_TOP_K,
)

RAG_QUERY_TEMPLATE = """
User topic: {query}

Detected categories:
{categories}

Use the detected categories as additional context,
but prioritize semantic relevance to the user's topic.
""".strip()


def classify_user_query(query):
    print("\n" + "=" * 60)
    print("STEP 1 - BERT CLASSIFICATION")
    print("=" * 60)

    print(f"\nUser topic: {query}")

    result = classify_article(query)

    print(f"\nPrimary category: {result['primary_category']}")

    print("\nAssigned categories:")

    for category in result["categories"]:
        print(f"  - {category}")

    print("\nTop confidences:")

    sorted_confidences = sorted(
        result["confidence"].items(),
        key=lambda x: x[1],
        reverse=True
    )

    for category, confidence in sorted_confidences[:5]:
        print(
            f"  {category:22s}: "
            f"{confidence:.2%}"
        )

    return result


def run_rag(query, classification):
    print("\n" + "=" * 60)
    print("STEP 2 - RAG")
    print("=" * 60)

    documents = load_top_trend_documents(
        TOP_TRENDS_FILE,
        CLEANED_DATA_FILE
    )

    if not documents:
        raise RuntimeError(
            "No articles were selected from Top Trends."
        )

    rag = MultiQueryRAG(documents)

    categories = ", ".join(
        classification["categories"]
    )

    rag_query = RAG_QUERY_TEMPLATE.format(
        query=query,
        categories=categories
    )

    print("\nRAG query:")
    print(rag_query)

    results, context = rag.build_context(
        rag_query,
        final_top_k=FINAL_TOP_K
    )

    print("\n" + "=" * 60)
    print("TOP RAG RESULTS")
    print("=" * 60)

    if not results:
        print("No relevant documents found.")

        return {
            "results": [],
            "context": "",
        }

    for i, result in enumerate(results, start=1):
        print(f"\n{i}. {result['title']}")

        print(
            f"   Trend: "
            f"{result.get('trend', '')}"
        )

        print(
            f"   Trend Score: "
            f"{result.get('trend_score', 0)}"
        )

        print(
            f"   Source: "
            f"{result.get('source', '')}"
        )

        print(
            f"   Category: "
            f"{result.get('category_hint', '')}"
        )

        print(
            f"   Similarity: "
            f"{result.get('similarity', 0):.4f}"
        )

    return {
        "results": results,
        "context": context,
    }


def generate_audio_for_articles(results):
    print("\n" + "=" * 60)
    print("STEP 3 - TEXT TO SPEECH")
    print("=" * 60)

    if not results:
        print("\nNo articles available for TTS.")
        return

    for i, article in enumerate(results, start=1):
        title = article.get("title", "").strip()
        content = article.get("content", "").strip()

        if not content:
            content = title

        if not content:
            print(
                f"\n[WARNING] Article {i} has no text."
            )
            continue

        text = f"{title}. {content}"

        output_path = (
            PROJECT_ROOT
            / "outputs"
            / f"article_{i}.mp3"
        )

        print(f"\nArticle {i}: {title}")

        text_to_speech(
            text=text,
            output_path=str(output_path)
        )

    print(
        "\n✓ All trending articles converted to audio."
    )


def main():
    print("=" * 60)
    print("TrendTales AI")
    print("BERT + Top Trends + RAG + TTS")
    print("=" * 60)

    query = input(
        "\nEnter your topic: "
    ).strip()

    if not query:
        print("\n[error] Topic cannot be empty.")
        return

    try:
        classification = classify_user_query(query)

        rag_output = run_rag(
            query,
            classification
        )

        generate_audio_for_articles(
            rag_output["results"]
        )

        print("\n" + "=" * 60)
        print("PIPELINE STATUS")
        print("=" * 60)

        print("\n✓ User query received")
        print("✓ BERT classification completed")
        print("✓ Top Trends loaded")
        print("✓ RAG retrieval completed")

        print(
            f"✓ Retrieved "
            f"{len(rag_output['results'])} "
            f"documents"
        )

        print("✓ Text-to-Speech completed")

        print("\nAudio files are available in:")
        print(f"  {PROJECT_ROOT / 'outputs'}")

        print("\n" + "=" * 60)
        print("RAG CONTEXT READY")
        print("=" * 60)

        print(rag_output["context"])

    except Exception as e:
        print("\n[ERROR]")
        print(str(e))
        raise


if __name__ == "__main__":
    main()