import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Contract Analyzer",
    page_icon="⚖️",
    layout="wide"
)

# --- Helpers ---

RISK_BADGE = {
    "high":   "🔴 HIGH",
    "medium": "🟡 MEDIUM",
    "low":    "🟢 LOW",
    "none":   "⚪ NONE",
}

RISK_COLOR = {
    "high":   "#ff4b4b",
    "medium": "#ffa500",
    "low":    "#21c354",
    "none":   "#cccccc",
}

def badge(level: str) -> str:
    color = RISK_COLOR.get(level, "#ccc")
    label = RISK_BADGE.get(level, level.upper())
    return f'<span style="background:{color};color:white;padding:2px 10px;border-radius:12px;font-size:0.8em;font-weight:bold">{label}</span>'


# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    groq_key = st.secrets.get("GROQ_API_KEY", "")
    
    if groq_key:
        st.success("✅ Groq API key loaded")
    else:
        groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
        if not groq_key:
            st.warning("⚠️ API key required")
    st.divider()
    st.markdown("**Pipeline**")
    st.markdown("1. 🔧 Pre-processing")
    st.markdown("2. ✂️ Segmentation")
    st.markdown("3. 🏷️ Classification")
    st.markdown("4. 🔍 RAG Search")
    st.markdown("5. ⚠️ Risk Scoring")
    st.markdown("6. 🔀 Conflict Detection")
    st.markdown("7. 💬 Negotiation")
    st.markdown("8. 📋 Summary")
    st.divider()
    st.caption("Run `build_cuad_index.py` once before first use.")


# --- Header ---
st.title("⚖️ Contract Analyzer")
st.caption("AI-powered contract risk analysis — partnership, service & commercial agreements")
st.divider()

# --- Upload ---
uploaded_file = st.file_uploader(
    "Upload a contract to analyze",
    type=["pdf", "docx", "txt"],
    label_visibility="collapsed"
)

if not uploaded_file:
    st.info("👆 Upload a PDF, DOCX, or TXT contract to get started.")
    st.stop()

st.success(f"📄 `{uploaded_file.name}` — {uploaded_file.size / 1024:.1f} KB")

if not st.button("🚀 Analyze Contract", type="primary", use_container_width=True):
    st.stop()

# --- Run Pipeline ---
from preprocessor import preprocess
from segmenter import segment
from classifier import classify
from rag import run_rag
from risk_scorer import score_risks
from conflict_detector import detect_conflicts
from negotiator import suggest
from summarizer import summarize

progress = st.progress(0, text="Starting pipeline...")

try:
    # Step 1
    progress.progress(10, text="🔧 Pre-processing document...")
    doc = preprocess(uploaded_file.read(), uploaded_file.name)

    if doc.doc_type == "UNRECOGNIZED":
        st.warning("⚠️ This document doesn't look like a commercial contract. Results may be inaccurate.")

    # Step 2
    progress.progress(20, text="✂️ Segmenting clauses...")
    clauses = segment(doc.raw_text)

    # Step 3
    progress.progress(35, text="🏷️ Classifying clauses...")
    clauses = classify(clauses, groq_api_key=groq_key)

    # Step 4
    progress.progress(50, text="🔍 Running RAG semantic search...")
    try:
        clauses = run_rag(clauses)
    except FileNotFoundError:
        st.warning("⚠️ FAISS index not found — skipping RAG. Run `build_cuad_index.py` first.")

    # Step 5
    progress.progress(65, text="⚠️ Scoring risks...")
    risk_scores = score_risks(clauses, groq_api_key=groq_key)

    # Step 6
    progress.progress(75, text="🔀 Detecting conflicts...")
    conflicts = detect_conflicts(clauses, groq_api_key=groq_key)

    # Step 7
    progress.progress(85, text="💬 Generating negotiation suggestions...")
    suggestions = suggest(clauses, risk_scores, groq_api_key=groq_key)

    # Step 8
    progress.progress(95, text="📋 Generating summaries...")
    business_summary, legal_summary = summarize(
        doc_type=doc.doc_type,
        clauses=clauses,
        risk_scores=risk_scores,
        conflicts=conflicts,
        suggestions=suggestions,
        groq_api_key=groq_key
    )

    progress.progress(100, text="✅ Analysis complete!")

except Exception as e:
    st.error(f"Pipeline error: {e}")
    st.stop()


# --- Results ---
st.divider()

# Quick stats row
risk_map = {r.clause_id: r for r in risk_scores}
high_count   = sum(1 for r in risk_scores if r.overall_risk.value == "high")
medium_count = sum(1 for r in risk_scores if r.overall_risk.value == "medium")
low_count    = sum(1 for r in risk_scores if r.overall_risk.value == "low")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("📄 Clauses",    len(clauses))
col2.metric("🔴 High Risk",  high_count)
col3.metric("🟡 Medium Risk", medium_count)
col4.metric("🟢 Low Risk",   low_count)
col5.metric("🔀 Conflicts",  len(conflicts))

st.divider()

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Clauses", "🔀 Conflicts", "💬 Suggestions", "📋 Business Summary", "⚖️ Legal Summary"
])


# Tab 1 — Clause table with risk badges
with tab1:
    st.subheader("Clause Analysis")
    for clause in clauses:
        risk = risk_map.get(clause.clause_id)
        overall = risk.overall_risk.value if risk else "none"
        dev = f"{clause.deviation_score:.2f}" if clause.deviation_score is not None else "N/A"
        match = clause.match_quality if hasattr(clause, "match_quality") and clause.match_quality else "N/A"

        with st.expander(f"{clause.heading or clause.clause_id}  —  {RISK_BADGE.get(overall, overall)}"):
            col1, col2, col3, col4 = st.columns(4)
            col1.markdown(f"**Type**  \n`{clause.clause_type.value if clause.clause_type else 'unknown'}`")
            col2.markdown(f"**Overall Risk**  \n{badge(overall)}", unsafe_allow_html=True)
            col3.markdown(f"**Deviation**  \n`{dev}`")
            col4.markdown(f"**Match Quality**  \n`{match}`")

            st.markdown("**Clause Text**")
            st.markdown(f"> {clause.text}")

            if risk and risk.reasoning:
                st.markdown("**Risk Reasoning**")
                col_l, col_f, col_o = st.columns(3)
                col_l.markdown(f"Legal: {badge(risk.legal_risk.value)}", unsafe_allow_html=True)
                col_f.markdown(f"Financial: {badge(risk.financial_risk.value)}", unsafe_allow_html=True)
                col_o.markdown(f"Operational: {badge(risk.operational_risk.value)}", unsafe_allow_html=True)
                st.caption(risk.reasoning)

            if hasattr(clause, "top_reference") and clause.top_reference:
                st.markdown("**Closest Reference Clause**")
                st.caption(clause.top_reference)


# Tab 2 — Conflicts
with tab2:
    st.subheader("Cross-Clause Conflicts")
    if not conflicts:
        st.success("✅ No conflicts detected.")
    else:
        for c in conflicts:
            clause_a = next((cl for cl in clauses if cl.clause_id == c.clause_id_a), None)
            clause_b = next((cl for cl in clauses if cl.clause_id == c.clause_id_b), None)
            heading_a = clause_a.heading if clause_a else c.clause_id_a
            heading_b = clause_b.heading if clause_b else c.clause_id_b

            # ✅ Plain text label for expander (no HTML)
            severity_label = RISK_BADGE.get(c.severity.value, c.severity.value.upper())
            with st.expander(f"{severity_label}  {heading_a}  ↔  {heading_b}", expanded=True):
                # ✅ Render badge with HTML inside the expander
                st.markdown(f"{badge(c.severity.value)} **{heading_a} ↔ {heading_b}**", unsafe_allow_html=True)
                st.markdown(f"**{c.conflict_description}**")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**{heading_a}**")
                    st.caption(clause_a.text[:300] if clause_a else "")
                with col2:
                    st.markdown(f"**{heading_b}**")
                    st.caption(clause_b.text[:300] if clause_b else "")


# Tab 3 — Negotiation Suggestions
with tab3:
    st.subheader("Negotiation Suggestions")
    if not suggestions:
        st.info("No suggestions generated — either no risky clauses or no API key provided.")
    else:
        for s in suggestions:
            clause = next((c for c in clauses if c.clause_id == s.clause_id), None)
            heading = clause.heading if clause else s.clause_id
            with st.expander(f"💬 {heading}"):
                st.markdown("**Original Clause**")
                st.markdown(f"> {s.original_text}")
                st.markdown("**Suggested Revision**")
                st.success(s.suggested_text)
                st.caption(f"💡 {s.rationale}")


# Tab 4 — Business Summary
with tab4:
    st.subheader("Business Summary")
    st.markdown(f"**Document Type:** `{doc.doc_type}` &nbsp;|&nbsp; **Total Clauses:** `{len(clauses)}`", unsafe_allow_html=True)
    st.divider()
    st.markdown(business_summary)


# Tab 5 — Legal Summary
with tab5:
    st.subheader("Legal Summary")
    st.markdown(f"**Document Type:** `{doc.doc_type}` &nbsp;|&nbsp; **Total Clauses:** `{len(clauses)}`", unsafe_allow_html=True)
    st.divider()
    st.markdown(legal_summary)