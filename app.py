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

        st.session_state[
            "articles"
        ] = results

        st.session_state[
            "article_index"
        ] = 0

        st.session_state[
            "search_query"
        ] = query

        st.rerun()

    except Exception as e:

        st.error(
            f"Application error: {e}"
        )

        st.stop()


if "articles" in st.session_state:

    articles = st.session_state[
        "articles"
    ]

    index = st.session_state[
        "article_index"
    ]

    article = articles[index]

    title = article.get(
        "title",
        "Untitled Article"
    )

    date = article.get(
        "date",
        ""
    )

    content = article.get(
        "content",
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

    st.write(content)

    st.divider()

    output_dir = (
        PROJECT_ROOT / "outputs"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    audio_text = (
        f"{title}. {content}"
    )

    audio_hash = hashlib.md5(
        audio_text.encode("utf-8")
    ).hexdigest()

    audio_path = (
        output_dir
        / f"article_{audio_hash}.mp3"
    )

    if audio_path.exists():

        st.audio(
            str(audio_path),
            format="audio/mp3"
        )

    else:

        if st.button(
            "Generate Audio",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Generating article voice..."
                ):

                    text_to_speech(
                        text=audio_text,
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

            st.rerun()

    if index == len(articles) - 1:

        st.info(
            "You have reached the end of the available articles."
        )