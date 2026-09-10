import sys
from pathlib import Path

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

from story_generator import generate_story

from tts import text_to_speech


RAG_QUERY_TEMPLATE = """
User topic: {query}

Detected categories:
{categories}

Use the detected categories as additional context,
but prioritize semantic relevance to the user's topic.
""".strip()


# =========================================================
# STEP 1 - BERT CLASSIFICATION
# =========================================================

def classify_user_query(query):

    print("\n" + "=" * 60)
    print("STEP 1 - BERT CLASSIFICATION")
    print("=" * 60)

    print(f"\nUser topic: {query}")

    result = classify_article(query)

    print(
        f"\nPrimary category: "
        f"{result['primary_category']}"
    )

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


# =========================================================
# STEP 2 - RAG
# =========================================================

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

        print(
            f"\n{i}. {result['title']}"
        )

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


# =========================================================
# STEP 3 - STORY GENERATION
# =========================================================

def generate_stories_for_articles(results):

    print("\n" + "=" * 60)
    print("STEP 3 - STORY GENERATION")
    print("=" * 60)

    if not results:

        print("\nNo articles available for story generation.")

        return []

    generated_results = []

    for i, article in enumerate(results, start=1):

        title = article.get(
            "title",
            "Untitled Article"
        ).strip()

        # RAG stores the retrieved article text in "text"
        text = article.get(
            "text",
            ""
        ).strip()

        if not text:

            print(
                f"\n[WARNING] Article {i} has no text."
            )

            continue

        print(
            f"\nGenerating story for Article {i}:"
        )

        print(
            f"Title: {title}"
        )

        try:

            story = generate_story(
                title,
                text,
                max_new_tokens=500
            )

            article_with_story = dict(article)

            article_with_story["story"] = story

            generated_results.append(
                article_with_story
            )

            print(
                "\n✓ Story generated successfully."
            )

            print("\nStory:")
            print(story)

        except Exception as e:

            print(
                f"\n[WARNING] "
                f"Story generation failed for Article {i}: {e}"
            )

    print(
        f"\n✓ Generated stories for "
        f"{len(generated_results)} articles."
    )

    return generated_results


# =========================================================
# STEP 4 - TEXT TO SPEECH
# =========================================================

def generate_audio_for_stories(results):

    print("\n" + "=" * 60)
    print("STEP 4 - TEXT TO SPEECH")
    print("=" * 60)

    if not results:

        print(
            "\nNo stories available for TTS."
        )

        return

    output_dir = (
        PROJECT_ROOT / "outputs"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for i, article in enumerate(
        results,
        start=1
    ):

        title = article.get(
            "title",
            ""
        ).strip()

        story = article.get(
            "story",
            ""
        ).strip()

        if not story:

            print(
                f"\n[WARNING] "
                f"Article {i} has no generated story."
            )

            continue

        output_path = (
            output_dir
            / f"story_{i}.mp3"
        )

        print(
            f"\nArticle {i}: {title}"
        )

        print(
            "Converting story to audio..."
        )

        text_to_speech(
            text=story,
            output_path=str(output_path)
        )

        print(
            f"✓ Audio saved: {output_path}"
        )

    print(
        "\n✓ All generated stories converted to audio."
    )


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():

    print("=" * 60)
    print("TrendTales AI")
    print(
        "BERT + Top Trends + RAG + "
        "Story Generation + TTS"
    )
    print("=" * 60)

    query = input(
        "\nEnter your topic: "
    ).strip()

    if not query:

        print(
            "\n[error] Topic cannot be empty."
        )

        return

    try:

        # STEP 1
        classification = classify_user_query(
            query
        )


        # STEP 2
        rag_output = run_rag(
            query,
            classification
        )


        if not rag_output["results"]:

            print(
                "\nNo articles were retrieved."
            )

            return


        # STEP 3
        story_results = generate_stories_for_articles(
            rag_output["results"]
        )


        # STEP 4
        generate_audio_for_stories(
            story_results
        )


        # =================================================
        # PIPELINE STATUS
        # =================================================

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

        print(
            f"✓ Generated "
            f"{len(story_results)} "
            f"stories"
        )

        print(
            "✓ Text-to-Speech completed"
        )

        print("\nAudio files are available in:")

        print(
            f"  {PROJECT_ROOT / 'outputs'}"
        )


        # =================================================
        # RAG CONTEXT
        # =================================================

        print("\n" + "=" * 60)
        print("RAG CONTEXT READY")
        print("=" * 60)

        print(
            rag_output["context"]
        )


    except Exception as e:

        print("\n[ERROR]")
        print(str(e))

        raise


if __name__ == "__main__":
    main()
