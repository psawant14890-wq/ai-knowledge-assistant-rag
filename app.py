import json
import os
import uuid

import streamlit as st

from app_graph import KnowledgeAssistant
from ingestion import IngestionError, KnowledgeIngestor


st.set_page_config(page_title="Knowledge Assistant", page_icon=":sparkles:", layout="wide")

st.markdown(
    """
    <style>
        :root {
            --bg: #0b1020;
            --panel: rgba(15, 23, 42, 0.82);
            --panel-strong: rgba(17, 24, 39, 0.96);
            --line: rgba(148, 163, 184, 0.16);
            --text: #edf2ff;
            --muted: #a5b4d4;
            --accent: #7c3aed;
            --accent-2: #22c55e;
            --accent-3: #38bdf8;
            --danger: #f87171;
        }
        .stApp {
            background:
                radial-gradient(circle at 0% 0%, rgba(124, 58, 237, 0.24), transparent 28%),
                radial-gradient(circle at 100% 10%, rgba(56, 189, 248, 0.18), transparent 30%),
                linear-gradient(180deg, #050816 0%, #0b1020 52%, #0e1428 100%);
            color: var(--text);
        }
        [data-testid="stSidebar"] {
            background: rgba(7, 10, 20, 0.95);
            border-right: 1px solid var(--line);
        }
        .block-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1.15rem 1.2rem;
            box-shadow: 0 24px 70px rgba(2, 6, 23, 0.35);
            backdrop-filter: blur(14px);
        }
        .hero-card {
            background:
                linear-gradient(135deg, rgba(124, 58, 237, 0.20), rgba(56, 189, 248, 0.10)),
                rgba(9, 14, 28, 0.88);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 28px;
            padding: 1.55rem 1.6rem;
            box-shadow: 0 28px 80px rgba(2, 6, 23, 0.42);
            margin-bottom: 1rem;
        }
        .eyebrow {
            display: inline-block;
            padding: 0.32rem 0.72rem;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.04);
            color: #d8def5;
            font-size: 0.78rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.9rem;
        }
        .hero-title {
            font-size: 2.65rem;
            font-weight: 700;
            color: #f8fbff;
            line-height: 1.05;
            margin-bottom: 0.55rem;
        }
        .hero-subtitle {
            color: #b8c4e5;
            font-size: 1rem;
            line-height: 1.65;
            max-width: 54rem;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 1rem;
        }
        .metric-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 0.95rem 1rem;
        }
        .metric-label {
            color: #9fb0d6;
            font-size: 0.82rem;
            margin-bottom: 0.35rem;
        }
        .metric-value {
            color: white;
            font-size: 1.45rem;
            font-weight: 700;
        }
        .soft-label {
            color: #9fb0d6;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.76rem;
            margin-bottom: 0.65rem;
        }
        .source-chip {
            display: inline-block;
            padding: 0.36rem 0.72rem;
            margin: 0 0.45rem 0.45rem 0;
            border-radius: 999px;
            background: rgba(124, 58, 237, 0.16);
            color: #e5d9ff;
            border: 1px solid rgba(167, 139, 250, 0.2);
            font-size: 0.86rem;
        }
        .status-pill {
            display: inline-block;
            padding: 0.32rem 0.72rem;
            border-radius: 999px;
            font-size: 0.8rem;
            margin-right: 0.5rem;
            border: 1px solid rgba(255,255,255,0.09);
        }
        .status-ready {
            background: rgba(34, 197, 94, 0.14);
            color: #bbf7d0;
        }
        .status-idle {
            background: rgba(148, 163, 184, 0.12);
            color: #dbe4ff;
        }
        .status-warn {
            background: rgba(248, 113, 113, 0.12);
            color: #fecaca;
        }
        .suggestion-card {
            padding: 0.85rem 0.95rem;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.03);
            color: #e7ecff;
            min-height: 96px;
        }
        .summary-box {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 0.95rem 1rem;
        }
        .citation-box {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 0.85rem 0.9rem;
            margin-bottom: 0.65rem;
        }
        .stDownloadButton button, .stButton button {
            border-radius: 14px !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_MESSAGE = (
    "Upload PDFs or website URLs, initialize the workspace, and then ask anything about your sources."
)
READY_MESSAGE = "Knowledge base initialized. Ask about the uploaded PDFs, compare sources, or generate summaries."
CLEARED_MESSAGE = "Chat history cleared. Your knowledge base is still loaded."


def bootstrap_state():
    defaults = {
        "messages": [{"role": "assistant", "content": DEFAULT_MESSAGE}],
        "assistant": None,
        "knowledge_summary": "",
        "chat_summary": "",
        "source_overview": [],
        "source_stats": None,
        "suggestions": [],
        "warnings": [],
        "answer_style": "Balanced",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def normalize_urls(raw_text):
    return [line.strip() for line in raw_text.splitlines() if line.strip()]


def resolve_api_key(manual_key):
    if manual_key:
        return manual_key.strip()

    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key

    try:
        return st.secrets.get("OPENAI_API_KEY", "").strip()
    except Exception:
        return ""


def initialize_workspace(pdf_files, urls, api_key):
    collection_name = f"knowledge-{uuid.uuid4().hex[:8]}"
    ingestor = KnowledgeIngestor(
        pdfs=pdf_files,
        urls=urls,
        api_key=api_key,
        persist_directory=f"data/chroma/{collection_name}",
        collection_name=collection_name,
    )
    assistant = KnowledgeAssistant(api_key=api_key, ingestor=ingestor)

    st.session_state.assistant = assistant
    st.session_state.source_overview = ingestor.get_source_overview()
    st.session_state.source_stats = ingestor.get_stats()
    st.session_state.suggestions = assistant.suggest_questions()
    st.session_state.knowledge_summary = ""
    st.session_state.chat_summary = ""
    st.session_state.warnings = ingestor.warnings
    st.session_state.messages = [{"role": "assistant", "content": READY_MESSAGE}]


def reset_chat():
    st.session_state.messages = [{"role": "assistant", "content": CLEARED_MESSAGE}]
    st.session_state.chat_summary = ""


def clear_workspace():
    st.session_state.assistant = None
    st.session_state.source_overview = []
    st.session_state.source_stats = None
    st.session_state.suggestions = []
    st.session_state.knowledge_summary = ""
    st.session_state.chat_summary = ""
    st.session_state.warnings = []
    st.session_state.messages = [{"role": "assistant", "content": DEFAULT_MESSAGE}]


def format_app_error(exc):
    text = str(exc)
    lowered = text.lower()
    if "insufficient_quota" in lowered or "429" in lowered:
        return (
            "OpenAI rejected the request because the API project has no available quota. "
            "Check billing, credits, or whether the key belongs to the correct API project."
        )
    if "401" in lowered or "invalid api key" in lowered:
        return "The OpenAI API key appears invalid. Check the key in your environment or Streamlit secrets."
    if "connection" in lowered or "timed out" in lowered:
        return "A network request failed while reading a website or contacting OpenAI. Try again in a moment."
    return text


def export_chat_markdown():
    lines = ["# Chat Transcript", ""]
    for message in st.session_state.messages:
        role = message["role"].title()
        lines.append(f"## {role}")
        lines.append(message["content"])
        lines.append("")
        if message.get("citations"):
            lines.append("Sources:")
            for citation in message["citations"]:
                lines.append(
                    f"- {citation['source']} | {citation['location']} | chunk {citation['chunk_index']}"
                )
            lines.append("")
    return "\n".join(lines)


def render_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        citations = message.get("citations", [])
        if citations:
            with st.expander("View sources", expanded=False):
                for citation in citations:
                    st.markdown(
                        f"""
                        <div class="citation-box">
                            <strong>{citation['source']}</strong><br>
                            <span style="color:#b8c4e5;">{citation['location']} | chunk {citation['chunk_index']}</span><br><br>
                            <span style="color:#e5e7eb;">{citation['snippet']}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


bootstrap_state()

stats = st.session_state.source_stats or {
    "sources": 0,
    "sections": 0,
    "chunks": 0,
    "pdf_sections": 0,
    "website_sections": 0,
}

st.markdown(
    f"""
    <div class="hero-card">
        <div class="eyebrow">Lovable-Inspired Workspace</div>
        <div class="hero-title">Search, chat, and summarize your knowledge base</div>
        <div class="hero-subtitle">
            Bring together PDFs and web pages into one polished research workspace. Ask grounded questions,
            review citations, generate summaries, and keep your conversation context intact.
        </div>
        <div style="margin-top:0.95rem;">
            <span class="status-pill {'status-ready' if st.session_state.assistant else 'status-idle'}">
                {'Knowledge base ready' if st.session_state.assistant else 'Waiting for sources'}
            </span>
            <span class="status-pill {'status-warn' if st.session_state.warnings else 'status-idle'}">
                {len(st.session_state.warnings)} warning(s)
            </span>
        </div>
        <div class="metric-grid">
            <div class="metric-card"><div class="metric-label">Sources</div><div class="metric-value">{stats['sources']}</div></div>
            <div class="metric-card"><div class="metric-label">Sections</div><div class="metric-value">{stats['sections']}</div></div>
            <div class="metric-card"><div class="metric-label">Chunks</div><div class="metric-value">{stats['chunks']}</div></div>
            <div class="metric-card"><div class="metric-label">Chat Turns</div><div class="metric-value">{max(len(st.session_state.messages) - 1, 0)}</div></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Setup")
    st.caption("Use local secrets or paste a key for this session.")
    manual_openai_key = st.text_input("OpenAI API Key", type="password")
    openai_key = resolve_api_key(manual_openai_key)
    if openai_key and not manual_openai_key:
        st.caption("Using API key from environment or Streamlit secrets.")

    pdf_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    website_input = st.text_area(
        "Website URLs",
        placeholder="https://example.com/article\nhttps://example.com/docs",
        help="Add one URL per line.",
    )
    urls = normalize_urls(website_input)

    st.markdown("### Answer Style")
    st.session_state.answer_style = st.selectbox(
        "Response style",
        options=["Balanced", "Precise", "Explainer", "Bullet Summary"],
        index=["Balanced", "Precise", "Explainer", "Bullet Summary"].index(
            st.session_state.answer_style
        ),
        label_visibility="collapsed",
    )

    if st.button("Initialize Knowledge Base", type="primary", use_container_width=True):
        if not openai_key:
            st.error("Enter an OpenAI API key or load one from secrets.")
        elif not pdf_files and not urls:
            st.error("Add at least one PDF or one valid website URL.")
        else:
            try:
                with st.spinner("Indexing sources and generating starter prompts..."):
                    initialize_workspace(pdf_files, urls, openai_key)
                st.success("Knowledge base ready.")
            except IngestionError as exc:
                st.error(format_app_error(exc))
            except Exception as exc:
                st.error(format_app_error(exc))

    control_col1, control_col2 = st.columns(2)
    with control_col1:
        if st.button("Reset Chat", use_container_width=True):
            reset_chat()
    with control_col2:
        if st.button("Clear All", use_container_width=True):
            clear_workspace()

    st.divider()
    st.markdown("### Generate")
    if st.button(
        "Summarize Knowledge Base",
        use_container_width=True,
        disabled=st.session_state.assistant is None,
    ):
        try:
            with st.spinner("Building knowledge summary..."):
                st.session_state.knowledge_summary = (
                    st.session_state.assistant.summarize_knowledge_base()
                )
        except Exception as exc:
            st.error(format_app_error(exc))

    if st.button(
        "Summarize Chat",
        use_container_width=True,
        disabled=st.session_state.assistant is None,
    ):
        try:
            with st.spinner("Summarizing conversation..."):
                st.session_state.chat_summary = st.session_state.assistant.summarize_chat(
                    st.session_state.messages
                )
        except Exception as exc:
            st.error(format_app_error(exc))

    if st.button(
        "Refresh Suggestions",
        use_container_width=True,
        disabled=st.session_state.assistant is None,
    ):
        try:
            with st.spinner("Refreshing suggestions..."):
                st.session_state.suggestions = st.session_state.assistant.suggest_questions()
        except Exception as exc:
            st.error(format_app_error(exc))

main_col, side_col = st.columns([1.5, 0.95], gap="large")

with main_col:
    chat_tab, summary_tab = st.tabs(["Chat", "Summaries"])

    with chat_tab:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.markdown('<div class="soft-label">Conversation</div>', unsafe_allow_html=True)
        for message in st.session_state.messages:
            render_message(message)

        if st.session_state.suggestions:
            st.markdown("#### Suggested prompts")
            suggestion_cols = st.columns(2)
            for index, suggestion in enumerate(st.session_state.suggestions[:4]):
                with suggestion_cols[index % 2]:
                    st.markdown(
                        f'<div class="suggestion-card">{suggestion}</div>',
                        unsafe_allow_html=True,
                    )

        question = st.chat_input(
            "Ask about a PDF, compare sources, or explore website content...",
            disabled=st.session_state.assistant is None,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if question:
            user_message = {"role": "user", "content": question}
            st.session_state.messages.append(user_message)
            render_message(user_message)

            try:
                with st.spinner("Thinking..."):
                    result = st.session_state.assistant.ask(
                        question,
                        st.session_state.messages[:-1],
                        answer_style=st.session_state.answer_style,
                    )
            except Exception as exc:
                st.error(format_app_error(exc))
            else:
                assistant_message = {
                    "role": "assistant",
                    "content": result["response"],
                    "citations": result["citations"],
                }
                st.session_state.messages.append(assistant_message)
                render_message(assistant_message)

    with summary_tab:
        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.markdown('<div class="summary-box">', unsafe_allow_html=True)
            st.markdown("#### Knowledge Summary")
            if st.session_state.knowledge_summary:
                st.write(st.session_state.knowledge_summary)
                st.download_button(
                    "Download Summary",
                    st.session_state.knowledge_summary,
                    file_name="knowledge-summary.txt",
                    use_container_width=True,
                )
            else:
                st.caption("Generate a knowledge summary from the sidebar.")
            st.markdown("</div>", unsafe_allow_html=True)
        with summary_col2:
            st.markdown('<div class="summary-box">', unsafe_allow_html=True)
            st.markdown("#### Chat Summary")
            if st.session_state.chat_summary:
                st.write(st.session_state.chat_summary)
                st.download_button(
                    "Download Chat Summary",
                    st.session_state.chat_summary,
                    file_name="chat-summary.txt",
                    use_container_width=True,
                )
            else:
                st.caption("Generate a chat summary after a few turns.")
            st.markdown("</div>", unsafe_allow_html=True)

with side_col:
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown('<div class="soft-label">Knowledge Base</div>', unsafe_allow_html=True)
    if st.session_state.source_overview:
        for item in st.session_state.source_overview:
            st.markdown(
                f"<span class='source-chip'>{item['source']} ({item['pages_or_sections']})</span>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No indexed sources yet.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="block-card" style="margin-top:1rem;">', unsafe_allow_html=True)
    st.markdown('<div class="soft-label">Source Stats</div>', unsafe_allow_html=True)
    st.json(stats, expanded=False)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="block-card" style="margin-top:1rem;">', unsafe_allow_html=True)
    st.markdown('<div class="soft-label">Workspace Tools</div>', unsafe_allow_html=True)
    st.download_button(
        "Download Chat Transcript",
        export_chat_markdown(),
        file_name="chat-transcript.md",
        use_container_width=True,
    )
    st.download_button(
        "Download Source Overview",
        json.dumps(st.session_state.source_overview, indent=2),
        file_name="source-overview.json",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.warnings:
        st.markdown('<div class="block-card" style="margin-top:1rem;">', unsafe_allow_html=True)
        st.markdown('<div class="soft-label">Warnings</div>', unsafe_allow_html=True)
        for warning in st.session_state.warnings:
            st.warning(warning)
        st.markdown("</div>", unsafe_allow_html=True)
