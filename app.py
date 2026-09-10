import sys
import os
import hashlib
from pathlib import Path

# Force UTF-8 encoding on standard output/error to prevent Windows charmap errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# pyrefly: ignore [missing-import]
import streamlit as st

# ── Project path setup ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bert_classifier.classify import classify_article
from rag.multi_query_rag import load_documents, MultiQueryRAG
from story_generation.story_generator import generate_story_from_events
from story_generation.tts import text_to_speech

# ── Categories option list ──────────────────────────────────────────────────
ALL_CATEGORIES = [
    "All",
    "Politics",
    "Economy",
    "Business",
    "Technology",
    "Climate",
    "Science",
    "Education",
    "International Affairs",
]

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrendTales AI",
    page_icon="🎧",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main-title-container {
        text-align: center;
        padding-top: 1rem;
        padding-bottom: 0.5rem;
    }
    
     .event-chip {
        display: inline-block;
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        color: #334155;
        border-radius: 8px;
        padding: 4px 10px;
        font-size: 0.82rem;
        font-weight: 500;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    .category-chip {
        display: inline-block;
        background: linear-gradient(135deg, #e0e7ff, #f3e8ff);
        border: 1px solid #a5b4fc;
        color: #4338ca;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 10px;
    }
    
    .story-text-container {
        background-color: #fafafa;
        border: 1px solid #eaeaea;
        border-radius: 10px;
        padding: 1.2rem;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #1f2937;
        white-space: pre-wrap;
    }
    
    .debug-box {
        background-color: #0f172a;
        color: #38bdf8;
        font-family: monospace;
        font-size: 0.85rem;
        padding: 1rem;
        border-radius: 8px;
        white-space: pre-wrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load RAG (cached resource) ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Initializing news engine...")
def load_rag_engine():
    docs = load_documents()
    return MultiQueryRAG(docs)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-title-container">
        <div class="main-title">🎧 TrendTales AI</div>
        <div class="subtitle">Turn trending news topics into engaging audio stories in seconds.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Optional Preferences (Expander in main view) ────────────────────────────
with st.expander("⚙️ Options & Filters", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        category_filter = st.selectbox(
            "Category Filter",
            options=ALL_CATEGORIES,
            index=0,
            help="Filter news by specific category",
        )
    with col2:
        top_k = st.slider(
            "Events Count",
            min_value=2,
            max_value=6,
            value=3,
            help="Number of news articles to reference in story",
        )

# ── Main Search Form ─────────────────────────────────────────────────────────
with st.form("search_form", clear_on_submit=False):
    query = st.text_input(
        "Topic Input",
        placeholder="Enter topic e.g. Artificial Intelligence, Renewable Energy, Economy...",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("✨ Generate Audio Story", use_container_width=True, type="primary")

# ── Pipeline Execution ───────────────────────────────────────────────────────
if submitted:
    if not query.strip():
        st.warning("Please enter a topic first.")
        st.stop()

    # Reset session state for new run
    for key in ["story", "events", "audio_path", "debug_logs", "detected_cats"]:
        st.session_state.pop(key, None)

    debug_logs = []
    
    # Use live status indicator for step-by-step feedback
    with st.status("⚙️ Processing pipeline steps...", expanded=True) as status:
        try:
            # Step 1: Classification
            st.write("🧠 **Step 1/4:** Multi-label classification with BERT model...")
            cls_result = classify_article(query.strip())
            primary_cat = cls_result["primary_category"]
            detected_cats = cls_result["categories"]
            debug_logs.append(f"[Step 1] BERT Multi-Label Categories: {detected_cats} (Primary: {primary_cat})")
            
            cats_formatted = ", ".join([f"`{c}`" for c in detected_cats])
            st.write(f"└ Multi-label categories detected: {cats_formatted}")

            # Step 2: RAG Retrieval
            st.write("🔍 **Step 2/4:** Retrieving trending news articles via FAISS RAG...")
            rag_category = None if category_filter == "All" else category_filter
            rag = load_rag_engine()
            events, _ = rag.build_context(
                query.strip(),
                category=rag_category or primary_cat,
                final_top_k=top_k,
            )

            if not events:
                status.update(label="❌ No relevant articles found.", state="error")
                st.error("No relevant news events found for this topic. Try another query.")
                st.stop()

            debug_logs.append(f"[Step 2] RAG Retrieved {len(events)} articles:")
            for idx, ev in enumerate(events, 1):
                debug_logs.append(f"  {idx}. {ev.get('title')} ({ev.get('source')})")
            st.write(f"└ Retrieved {len(events)} trending events.")

            # Step 3: Story Generation
            st.write("✍️ **Step 3/4:** Generating story with Qwen2.5-72B model...")
            story = generate_story_from_events(events)
            debug_logs.append(f"[Step 3] Story generated ({len(story)} characters).")
            st.write("└ Story written successfully.")

            # Step 4: Text to Speech
            st.write("🔊 **Step 4/4:** Synthesizing speech audio with gTTS...")
            output_dir = PROJECT_ROOT / "outputs"
            output_dir.mkdir(exist_ok=True)
            story_hash = hashlib.md5(story.encode("utf-8")).hexdigest()[:12]
            audio_path = output_dir / f"story_{story_hash}.mp3"

            if not audio_path.exists():
                text_to_speech(text=story, output_path=str(audio_path))

            debug_logs.append(f"[Step 4] Audio generated successfully: {audio_path.name}")
            st.write("└ Audio ready.")

            # Complete status bar
            status.update(label="✅ Audio Story Created Successfully!", state="complete", expanded=False)

            st.session_state["story"] = story
            st.session_state["events"] = events
            st.session_state["detected_cats"] = detected_cats
            st.session_state["audio_path"] = str(audio_path)
            st.session_state["story_hash"] = story_hash
            st.session_state["debug_logs"] = "\n".join(debug_logs)

        except Exception as e:
            status.update(label="❌ Pipeline Error Occurred", state="error", expanded=True)
            st.error(f"Failed to generate story: {e}")
            st.stop()

# ── Display Audio Output & Story/Debug Views ──────────────────────────────────
if "audio_path" in st.session_state and Path(st.session_state["audio_path"]).exists():
    # Render Multi-Label Categories Badges
    if "detected_cats" in st.session_state:
        st.markdown("**🏷️ Detected Multi-Label Categories (BERT):**")
        cat_chips = "".join(
            f'<span class="category-chip">🏷️ {cat}</span>'
            for cat in st.session_state["detected_cats"]
        )
        st.markdown(cat_chips, unsafe_allow_html=True)

    st.markdown("### 🔊 Listen to Story")
    st.audio(st.session_state["audio_path"], format="audio/mp3")

    with open(st.session_state["audio_path"], "rb") as f:
        st.download_button(
            label="⬇️ Download Audio (MP3)",
            data=f,
            file_name=f"trendtale_{st.session_state.get('story_hash', 'audio')}.mp3",
            mime="audio/mp3",
            use_container_width=True,
        )

    # Referenced news events chips
    if "events" in st.session_state:
        st.markdown("**Referenced News Events:**")
        events_html = "".join(
            f'<span class="event-chip">📰 {ev.get("title", "Untitled")}</span>'
            for ev in st.session_state["events"]
        )
        st.markdown(events_html, unsafe_allow_html=True)

    st.markdown("---")

    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        with st.expander("📜 Show Written Story (Text)", expanded=False):
            st.markdown(
                f'<div class="story-text-container">{st.session_state.get("story", "")}</div>',
                unsafe_allow_html=True,
            )

    with col_btn2:
        with st.expander("🐛 Debug Pipeline Logs", expanded=False):
            st.markdown(
                f'<div class="debug-box">{st.session_state.get("debug_logs", "No logs available.")}</div>',
                unsafe_allow_html=True,
            )