import sys
import hashlib
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from bert_classifier.classify import classify_article

from rag import (
    load_documents,
    MultiQueryRAG,
    CLEANED_DATA_FILE,
)

from story_generator import generate_story

from tts import text_to_speech


st.set_page_config(
    page_title="TrendTales AI",
    page_icon="📰",
    layout="wide"
)


st.title("TrendTales AI")


query = st.text_input(
    "Enter your topic",
    placeholder="e.g. Artificial Intelligence"
)


search_button = st.button(
    "Search",
    type="primary",
    use_container_width=True
)


@st.cache_resource
def load_rag():

    documents = load_documents(
        CLEANED_DATA_FILE
    )

    if not documents:
        raise RuntimeError(
            "No articles were found in cleaned_data.json."
        )

    return MultiQueryRAG(
        documents
    )


if search_button:

    if not query.strip():

        st.warning(
            "Please enter a topic."
        )

        st.stop()

    try:

        with st.spinner(
            "Searching through the news dataset..."
        ):

            classification = classify_article(
                query
            )

            rag = load_rag()

            categories = ", ".join(
                classification.get(
                    "categories",
                    []
                )
            )

            rag_query = f"""
User topic:
{query}

Detected categories:
{categories}

Find news articles that are semantically relevant
to the user's topic.
""".strip()

            results, _ = rag.build_context(
                rag_query,
                final_top_k=None
            )

        if not results:

            st.warning(
                "No relevant articles were found."
            )

            st.stop()

        st.session_state["articles"] = results

        st.session_state["article_index"] = 0

        st.session_state["search_query"] = query

        # Clear previous story
        st.session_state.pop("story", None)

        st.rerun()

    except Exception as e:

        st.error(
            f"Application error: {e}"
        )

        st.stop()


# =========================================================
# DISPLAY CURRENT ARTICLE
# =========================================================

if "articles" in st.session_state:

    articles = st.session_state["articles"]

    index = st.session_state["article_index"]

    article = articles[index]


    title = article.get(
        "title",
        "Untitled Article"
    )


    date = article.get(
        "date",
        ""
    )


    # RAG returns article text under "text"
    text = article.get(
        "text",
        ""
    ).strip()


    st.divider()


    st.caption(
        f"Article {index + 1} of {len(articles)}"
    )


    st.header(title)


    if date:

        st.caption(
            f"Published: {date}"
        )


    st.write(text)


    st.divider()


    # =====================================================
    # STORY GENERATION
    # =====================================================

    st.subheader("📖 Story")


    if st.button(
        "Generate Story",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Generating story with Qwen3..."
            ):

                story = generate_story(
                    title,
                    text,
                    max_new_tokens=500
                )


            st.session_state["story"] = story

            st.rerun()


        except Exception as e:

            st.error(
                f"Story generation failed: {e}"
            )


    # Display generated story

    if "story" in st.session_state:

        story = st.session_state["story"]

        st.write(story)


        # =================================================
        # TEXT TO SPEECH FOR STORY
        # =================================================

        st.divider()

        output_dir = (
            PROJECT_ROOT / "outputs"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        audio_hash = hashlib.md5(
            story.encode("utf-8")
        ).hexdigest()


        audio_path = (
            output_dir
            / f"story_{audio_hash}.mp3"
        )


        if audio_path.exists():

            st.audio(
                str(audio_path),
                format="audio/mp3"
            )

        else:

            if st.button(
                "Generate Story Audio",
                use_container_width=True
            ):

                try:

                    with st.spinner(
                        "Generating story voice..."
                    ):

                        text_to_speech(
                            text=story,
                            output_path=str(
                                audio_path
                            )
                        )

                    st.rerun()

                except Exception as e:

                    st.error(
                        f"TTS failed: {e}"
                    )


    st.divider()


    # =====================================================
    # PREVIOUS / NEXT
    # =====================================================

    previous_col, counter_col, next_col = st.columns(
        [1, 1, 1]
    )


    with previous_col:

        if st.button(
            "← Previous",
            disabled=index == 0,
            use_container_width=True
        ):

            st.session_state[
                "article_index"
            ] -= 1

            st.session_state.pop(
                "story",
                None
            )

            st.rerun()


    with counter_col:

        st.markdown(
            f"""
            <div style="
                text-align:center;
                padding-top:8px;
                font-weight:bold;
            ">
                {index + 1} / {len(articles)}
            </div>
            """,
            unsafe_allow_html=True
        )


    with next_col:

        if st.button(
            "Next →",
            disabled=index == len(articles) - 1,
            use_container_width=True
        ):

            st.session_state[
                "article_index"
            ] += 1

            st.session_state.pop(
                "story",
                None
            )

            st.rerun()


    if index == len(articles) - 1:

        st.info(
            "You have reached the end of the available articles."
        )
