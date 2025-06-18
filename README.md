```markdown
# Horizon-Scanning Agent 🛰️  

Python prototype that **automatically scans open sources, enriches them with NLP,
stores semantic embeddings, and answers policy questions with an LLM**  
– initially tuned for labour-market and social-policy signals, but easily retargeted.

---

## ✨  What it does

| Stage | Purpose | Key libs |
|-------|---------|----------|
| **1. Harvest** | Async-download ~60 RSS / API feeds (ECB, Eurostat, NYT, Scopus, OpenAIRE, CORDIS, …) | `httpx`, `feedparser`, `xml.etree`, `tenacity` |
| **2. Clean & Filter** | De-duplicate, parse dates, keyword-filter “policy” terms | `pandas`, `re` |
| **3. Enrich** | TF-IDF keywords, recency labels, trend score, **PESTLE** zero-shot tags | `scikit-learn`, `transformers` |
| **4. Embed & Index** | MiniLM sentence embeddings → **FAISS** vector index (+ metadata parquet) | `sentence-transformers`, `faiss-cpu` |
| **5. Retrieve** | `faiss_search.py` returns top-*k* docs for a query |  |
| **6. Answer** | `agent_app.py` feeds snippets + question into OpenAI GPT-4o (or any compatible endpoint) | `openai` |

---

## 📦  Folder layout

```

.
├── policy\_signal\_scanner\_v3.py   ← main ETL/embedding pipeline
├── agent\_app.py                  ← chat-style Q\&A loop
├── retrieval\_backend.py          ← thin helper used by the agent
├── faiss\_search.py               ← 1-liner wrapper for quick tests
├── test\_query.py                 ← sanity-check script
├── sources.yaml                  ← list of RSS / API endpoints
├── keywords.yaml                 ← editable policy keyword list
├── cordis-h2020projects.json     ← tiny sample dataset (20 records)
├── requirements.txt
└── .gitignore

````

> **Large runtime artefacts** (`*.faiss`, `*.parquet`, `horizon_scanning_combined.csv`) are ignored by Git via `.gitignore`.

---

## 🚀  Quick start

```bash
# 1. clone
git clone https://github.com/<your-user>/horizon-scanning-agent.git
cd horizon-scanning-agent

# 2. install deps (ideally inside a virtualenv)
pip install -r requirements.txt

# 3. add your secrets
cp .env.example .env            # then edit
#   OPENAI_API_KEY=...
#   SCOPUS_API_KEY=...(optional)

# 4. run the pipeline  (~3-5 min on CPU)
python policy_signal_scanner_v3.py

# 5. chat with the agent
python agent_app.py
````

---

## ⚙️  Configuration

| File                          | What you can tweak                                                    |
| ----------------------------- | --------------------------------------------------------------------- |
| `sources.yaml`                | Comment / add RSS or API endpoints.                                   |
| `keywords.yaml`               | Add/remove policy terms to widen or narrow the filter.                |
| `.env`                        | `OPENAI_API_KEY`, optional `SCOPUS_API_KEY`, model endpoint URL, etc. |
| `policy_signal_scanner_v3.py` | `ENABLE_SUMMARY`, scoring half-life, batch sizes, etc.                |

All YAML and environment variables are loaded at runtime—no code edits required for normal tuning.

---

## 🧐  Testing retrieval only

```bash
python test_query.py "impact of labour mobility on EU policy"
```

Returns a GitHub-styled table with the five closest documents + distances.

---

## 🔒  Responsible & secure use

* Scraper fetches **open, public** sources only and respects polite retry/back-off.
* Secrets live in **`.env`** and are never committed.
* Model prompts and top-k snippets are logged to stdout for auditability.

---

## 📝  License

MIT – see `LICENSE`.

Dal Borgo R. (2025) Horizon-Scanning Agent, v0.1.*

```
```
