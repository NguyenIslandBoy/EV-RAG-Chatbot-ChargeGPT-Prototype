"""
streamlit_app.py
================
ChargeGPT — Streamlit web interface.

Run:
    streamlit run streamlit_app.py
"""

import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="ChargeGPT",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling — clean light theme, no background overrides
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}

#MainMenu, footer { visibility: hidden; }
.stDeployButton { display: none; }

.chargegpt-logo {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #1a1a2e;
    margin-bottom: 0.1rem;
}
.chargegpt-logo span { color: #16a34a; }
.chargegpt-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #9ca3af;
    letter-spacing: 0.08em;
    margin-bottom: 1.5rem;
}

.msg-user {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 1rem;
}
.msg-user-bubble {
    background: #16a34a;
    color: #ffffff;
    padding: 0.65rem 1rem;
    border-radius: 14px 14px 2px 14px;
    max-width: 72%;
    font-size: 0.92rem;
    line-height: 1.55;
}
.msg-assistant {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 1rem;
}
.msg-assistant-bubble {
    background: #f8f9fa;
    border: 1px solid #e5e7eb;
    color: #1a1a2e;
    padding: 0.85rem 1.1rem;
    border-radius: 14px 14px 14px 2px;
    max-width: 82%;
    font-size: 0.92rem;
    line-height: 1.65;
}

.route-badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    margin-bottom: 0.5rem;
    font-weight: 700;
}
.badge-sql   { background: #dbeafe; color: #1d4ed8; border: 1px solid #bfdbfe; }
.badge-rag   { background: #ede9fe; color: #6d28d9; border: 1px solid #ddd6fe; }
.badge-both  { background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }
.badge-error { background: #fee2e2; color: #dc2626; border: 1px solid #fecaca; }

.sql-block {
    background: #1e293b;
    border-radius: 6px;
    padding: 0.65rem 0.9rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #7dd3fc;
    margin-top: 0.65rem;
    white-space: pre-wrap;
    word-break: break-all;
}

.latency-tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: #9ca3af;
    margin-left: 0.4rem;
}

.empty-state {
    text-align: center;
    padding: 4rem 0 2rem;
    color: #d1d5db;
}
.empty-icon { font-size: 2.5rem; margin-bottom: 0.75rem; }
.empty-text {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

[data-testid="stSidebar"] h3 {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #6b7280 !important;
}

.sidebar-footer {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: #9ca3af;
    letter-spacing: 0.05em;
    line-height: 1.8;
    margin-top: 1rem;
}

[data-testid="stFormSubmitButton"] > button {
    background-color: #16a34a !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    width: 100% !important;
    padding: 0.6rem !important;
}
[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #15803d !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Resource init (cached once for app lifetime)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def init_resources():
    import config  # noqa — triggers logging setup
    from db import get_connection
    from rag.vector_store import build_vector_store
    return get_connection(), build_vector_store()


def get_route_and_answer(question, db_con, vector_store):
    from engine.router import classify_query, ROUTE_SQL, ROUTE_RAG, ROUTE_BOTH
    from engine.query_engine import run_query
    from rag.rag_engine import run_rag_query
    from chat import merge_responses

    route = classify_query(question)

    if route == ROUTE_SQL:
        resp = run_query(question, db_con)
        return resp.answer, "sql", resp.sql
    elif route == ROUTE_RAG:
        resp = run_rag_query(question, vector_store)
        return resp.answer, "rag", None
    elif route == ROUTE_BOTH:
        sql_resp = run_query(question, db_con)
        rag_resp = run_rag_query(question, vector_store)
        if not sql_resp.success:
            return rag_resp.answer, "rag", None
        if not rag_resp.success:
            return sql_resp.answer, "sql", sql_resp.sql
        return merge_responses(sql_resp.answer, rag_resp.answer), "both", sql_resp.sql

    return "I couldn't determine how to handle that.", "error", None


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

for key, default in [
    ("messages", []),
    ("total_queries", 0),
    ("sql_queries", 0),
    ("rag_queries", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚡ ChargeGPT")
    st.divider()

    st.markdown("### Session stats")
    c1, c2 = st.columns(2)
    sql_pct = (
        round(st.session_state.sql_queries / st.session_state.total_queries * 100)
        if st.session_state.total_queries > 0 else 0
    )
    c1.metric("Queries", st.session_state.total_queries)
    c2.metric("SQL %", f"{sql_pct}%")

    st.divider()

    st.markdown("### Route legend")
    st.markdown(
        '<span class="route-badge badge-sql">SQL</span> &nbsp;Structured data query<br><br>'
        '<span class="route-badge badge-rag">RAG</span> &nbsp;Knowledge retrieval<br><br>'
        '<span class="route-badge badge-both">BOTH</span> &nbsp;Data + context',
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### Try asking")
    examples = [
        "How many sessions in total?",
        "What is the peak charging hour?",
        "What is driver laxity?",
        "Which site uses the most energy?",
        "How does pricing affect behaviour?",
        "What is the carbon intensity?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state["_prefill"] = ex

    st.divider()

    if st.button("🗑  Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.session_state.sql_queries = 0
        st.session_state.rag_queries = 0
        st.rerun()

    st.markdown(
        '<div class="sidebar-footer">'
        'ACN-DATA · 66,446 SESSIONS<br>'
        'CALTECH · JPL · OFFICE 1<br>'
        'phi3.5 · FAISS · DuckDB'
        '</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="chargegpt-logo">Charge<span>GPT</span></div>'
    '<div class="chargegpt-sub">EV CHARGING ANALYTICS ASSISTANT · ACN-DATA</div>',
    unsafe_allow_html=True,
)

# Chat history
if not st.session_state.messages:
    st.markdown(
        '<div class="empty-state">'
        '<div class="empty-icon">⚡</div>'
        '<div class="empty-text">Ask anything about the EV charging dataset</div>'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="msg-user">'
                f'<div class="msg-user-bubble">{msg["content"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            route = msg.get("route", "sql")
            latency = msg.get("latency", 0)
            latency_html = f'<span class="latency-tag">· {latency:.1f}s</span>' if latency else ""
            sql_html = (
                f'<div class="sql-block">{msg["sql"]}</div>'
                if msg.get("sql") else ""
            )
            content = msg["content"].replace("\n", "<br>")
            st.markdown(
                f'<div class="msg-assistant"><div class="msg-assistant-bubble">'
                f'<div><span class="route-badge badge-{route}">{route.upper()}</span>{latency_html}</div>'
                f'{content}{sql_html}'
                f'</div></div>',
                unsafe_allow_html=True,
            )

# Input form
st.divider()

prefill = st.session_state.pop("_prefill", "")

with st.form(key="chat_form", clear_on_submit=True):
    col_in, col_btn = st.columns([5, 1])
    with col_in:
        user_input = st.text_input(
            label="q",
            value=prefill,
            placeholder="Ask about sessions, energy, peak hours, charging behaviour...",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button("Send ⚡", use_container_width=True)

if submitted and user_input.strip():
    question = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Thinking..."):
        try:
            db_con, vector_store = init_resources()
            t0 = time.perf_counter()
            answer, route, sql = get_route_and_answer(question, db_con, vector_store)
            latency = time.perf_counter() - t0
        except Exception as e:
            answer, route, sql, latency = f"Error: {e}", "error", None, 0.0

    st.session_state.total_queries += 1
    if route in ("sql", "both"):
        st.session_state.sql_queries += 1
    if route in ("rag", "both"):
        st.session_state.rag_queries += 1

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "route": route,
        "sql": sql,
        "latency": latency,
    })

    st.rerun()