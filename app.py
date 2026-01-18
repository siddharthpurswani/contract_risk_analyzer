import streamlit as st
from main import run_contract_analysis

st.title("📄 Contract Risk Analyzer")

uploaded_file = st.file_uploader("Upload contract (.txt)", type=["txt"])

if uploaded_file:
    contract_text = uploaded_file.read().decode("utf-8")

    with st.spinner("Analyzing contract..."):
        result = run_contract_analysis(contract_text)

    st.subheader("Executive Summary")
    st.write(result["executive_summary"])

    st.subheader("Risk Summary")
    st.write(result["risk_summary"])
