# 📄 Contract Risk Analyzer

> **AI-powered contract intelligence** — upload any agreement and get instant risk scoring, conflict detection, and negotiation-ready clause rewrites.

🔗 **[Live Demo](https://contractriskanalyzer-snmjpw3fhcgtmbyialvtdq.streamlit.app/)** &nbsp;|&nbsp; 💻 **[GitHub](https://github.com/siddharthpurswani/contract_risk_analyzer)**

---

## 🧠 What It Does

Contract Risk Analyzer is an end-to-end pipeline that ingests PDF, DOCX, or TXT contracts and delivers structured risk intelligence across legal, financial, and operational dimensions — in seconds.

It combines **FAISS semantic search**, **14 static rule checks**, and **LLaMA 3.3-70B via Groq** in a hybrid detection architecture, benchmarking every uploaded clause against 13,000+ real legal clauses from the CUAD dataset.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📂 **Multi-format Ingestion** | Supports PDF, DOCX, and TXT contract files |
| ✂️ **Clause Segmentation** | Automatically splits agreements into discrete, analyzable clauses |
| 🔍 **RAG Semantic Search** | FAISS + MMR retrieval against 13,000+ CUAD clauses to detect deviation from standard legal language |
| ⚠️ **Hybrid Risk Scoring** | 14 static rules + LLaMA 3.3-70B (70B params) via Groq for legal, financial & operational risk |
| 🔗 **Cross-Clause Conflict Detection** | Flags jurisdiction mismatches, termination-liability contradictions, and more |
| ✍️ **Negotiation Suggestions** | LLM-generated clause rewrites that replace risky language with balanced alternatives |
| 📊 **Dual Summaries** | Separate executive (business) and legal-audience summaries |
| 🎨 **Visual UI** | Color-coded risk badges, conflict graph visualizations via Streamlit |

---

## 🏗️ Architecture

```
Upload (PDF / DOCX / TXT)
        │
        ▼
Clause Segmentation
        │
        ▼
┌───────────────────────────────────┐
│         Hybrid Risk Engine        │
│                                   │
│  FAISS Vector Search (CUAD 13k+) │
│  + 14 Static Rule Checks          │
│  + LLaMA 3.3-70B via Groq API     │
└───────────────────────────────────┘
        │
        ▼
Cross-Clause Conflict Detection
        │
        ▼
Risk Scoring (Legal / Financial / Operational)
        │
        ▼
Negotiation Suggestions + Dual Summaries
        │
        ▼
Streamlit UI (Risk Badges + Conflict Graph)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit |
| **LLM** | LLaMA 3.3-70B via Groq API |
| **Vector Search** | FAISS with MMR retrieval |
| **Dataset** | CUAD (Contract Understanding Atticus Dataset) — 13,000+ clauses |
| **File Parsing** | PyPDF2, python-docx |
| **Language** | Python 3.10+ |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/siddharthpurswani/contract_risk_analyzer.git
cd contract_risk_analyzer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> Get your free Groq API key at [console.groq.com](https://console.groq.com)

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📋 Usage

1. **Upload** a contract file (PDF, DOCX, or TXT)
2. **Wait** for the pipeline to segment and analyze clauses (~10–30s depending on contract length)
3. **Review** the risk dashboard:
   - 🔴 High / 🟡 Medium / 🟢 Low risk badges per clause
   - Conflict graph showing cross-clause contradictions
   - Deviation scores against CUAD standard language
4. **Export** negotiation suggestions and summaries for legal or executive review

---

## 📁 Project Structure

```
contract_risk_analyzer/
├── app.py                  # Streamlit entry point
├── pipeline/
│   ├── ingestion.py        # PDF/DOCX/TXT parsing
│   ├── segmentation.py     # Clause splitting logic
│   ├── vector_store.py     # FAISS index + MMR retrieval
│   ├── risk_engine.py      # Static rules + LLM classification
│   ├── conflict_detector.py# Cross-clause conflict logic
│   └── summarizer.py       # Dual summary generation
├── data/
│   └── cuad_clauses/       # CUAD dataset embeddings
├── requirements.txt
└── .env.example
```

---

## 🔬 How Risk Scoring Works

Each clause is evaluated across three dimensions:

- **Legal Risk** — unusual termination terms, unilateral amendment rights, jurisdiction conflicts
- **Financial Risk** — uncapped liabilities, payment penalties, indemnification scope
- **Operational Risk** — IP ownership ambiguity, data handling obligations, SLA enforceability

Risk is computed as a weighted combination of:
1. **Semantic deviation score** from CUAD standard clause embeddings (FAISS)
2. **Rule match count** across 14 static legal heuristics
3. **LLM confidence score** from LLaMA 3.3-70B classification

---

## 📜 Dataset

This project uses the **CUAD (Contract Understanding Atticus Dataset)** — a large-scale legal dataset containing 13,000+ expert-annotated contract clauses across 41 clause categories, curated by legal professionals.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 📬 Contact

**Siddharth Purswani** — [GitHub](https://github.com/siddharthpurswani)

---

## ⚠️ Disclaimer

This tool is intended for informational and educational purposes only. It does not constitute legal advice. Always consult a qualified legal professional before making decisions based on contract analysis.
