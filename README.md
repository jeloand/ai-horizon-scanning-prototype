# Policy Horizon Scanner 🛰️

*A Python toolkit that periodically (or on-demand) harvests labour-market & social-policy signals from open sources, enriches them with NLP, stores semantic embeddings in FAISS, and answers questions through a Retrieval-Augmented Generation (RAG) agent (CLI **and** Streamlit UI).*

*Although pre-configured for labour-market and social-policy content, the pipeline is domain-agnostic — swap the keyword list, adjust the API queries, and you’re ready for climate tech, healthcare innovation, or any other topic.*

---

## Table of Contents

1. [Key Capabilities](#key-capabilities)
2. [Folder Layout](#folder-layout)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Running the Pipeline](#running-the-pipeline)
6. [Chatting with the Agent](#chatting-with-the-agent)
7. [Streamlit Dashboard](#streamlit-dashboard)
8. [Testing](#testing)
9. [Outputs & Caching](#outputs--caching)
10. [Extending](#extending)
11. [Troubleshooting](#troubleshooting)
12. [License](#license)

---

## Key Capabilities

| #     | Stage          | What happens                                                                              | Main libs                                     |
| ----- | -------------- | ----------------------------------------------------------------------------------------- | --------------------------------------------- |
| **1** | Harvest        | Async-fetch \~60 RSS/APIs (ECB, Eurostat, Scopus, OpenAIRE, CORDIS …)                     | `httpx`, `feedparser`, `tenacity`             |
| **2** | Clean + Filter | De-dupe, infer dates, keep rows containing policy keywords                                | `pandas`, `re`, `PyYAML`                      |
| **3** | Enrich         | TF-IDF keywords, recency buckets, trend score, **PESTLE** zero-shot tags (cached to disk) | `scikit-learn`, `transformers`, `disk_cache`  |
| **4** | Embed + Index  | MiniLM sentence embeddings → **FAISS** vector index                                       | `sentence_transformers`, `faiss-cpu`          |
| **5** | Retrieve       | `retrieval_backend.snippets()` formats top-*k* snippets for a query                       | `faiss`, `numpy`, `pandas`                    |
| **6** | Answer         | `agent_app.py` feeds snippets + question to **GPT-4o**                                    | `openai`                                      |
| **7** | Visualise      | `app_streamlit.py` dashboard: KPIs, preview table, and RAG chat                           | `streamlit`                                   |
| **8** | Test           | `test_query.py` & `pytest` sanity-check retrieval quality                                 | `pytest`, `tabulate`                          |

---

## Folder Layout

```text
.
├── policy_signal_scanner_v3.py       # main ETL / embedding pipeline
├── retrieval_backend.py              # RAG helper used by the agent & UI
├── agent_app.py                      # CLI chat loop
├── app_streamlit.py                  # Streamlit dashboard + chat
├── faiss_search.py                   # thin wrapper for quick retrieval tests
├── test_query.py                     # pytest / CLI sanity test
├── disk_cache.py                     # JSON on-disk memoisation decorator
├── requirements.txt
├── sources.yaml                      # RSS / API endpoints
├── keywords.yaml                     # editable policy keyword list
├── cordis-h2020projects.json         # tiny sample dataset (optional)
├── LICENSE                           # MIT
└── .gitignore                        # ignores *.faiss, *.parquet, combined CSV
```

Large artefacts (`*.faiss`, `*.parquet`, `horizon_scanning_combined.csv`) are **not** committed.

---

## Quick Start

```bash
# 1) clone & enter
git clone https://github.com/<you>/policy-horizon-scanner.git
cd policy-horizon-scanner

# 2) create env & install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3) provide secrets
cp example.config2.json config2.json      # add your OpenAI key
export SCOPUS_API_KEY="your-scopus-key"   # optional but recommended

# 4) run the pipeline  (~2–5 min on CPU)
python policy_signal_scanner_v3.py

# 5) chat with the agent
python agent_app.py
```

---

## Configuration

| Item                | Where                                           | Notes                              |
| ------------------- | ----------------------------------------------- | ---------------------------------- |
| **OpenAI key**      | `config2.json` (`{"OPENAI_API_KEY": "sk-..."}`) | read by `agent_app.py` & Streamlit |
| **Scopus key**      | `SCOPUS_API_KEY` env-var (or edit constant)     | used in Stage 1                    |
| **Feeds**           | `sources.yaml`                                  | add / comment out as needed        |
| **Policy keywords** | `keywords.yaml`                                 | drives Stage 2 filter              |
| **Pipeline flags**  | top of `policy_signal_scanner_v3.py`            | e.g. `ENABLE_SUMMARY = True`       |

You can also use a `.env` file; `python-dotenv` is in the requirements.

---

## Running the Pipeline

```bash
python policy_signal_scanner_v3.py
```

Outputs
`horizon_scanning_combined.csv`, `horizon_scanning_meta.parquet`, `horizon_scanning.faiss` appear in the repo root. The default run (≈60 feeds) finishes in **< 5 min on a laptop CPU**.

---

## Chatting with the Agent

```bash
python agent_app.py
```

**Flow**

```
user question
   ↓
retrieval_backend.snippets()      # embeds query & pulls k-nearest docs
   ↓
prompt = context + question
   ↓
GPT-4o chat-completion
   ↓
console answer
```

Exit with `quit` / `q`.

---

## Streamlit Dashboard

```bash
streamlit run app_streamlit.py
```

* **Overview tab** – KPI cards, source mix bar-chart, preview table.
* **Ask GPT-4o tab** – same RAG workflow but inside the browser.

> The app checks for `.csv`, `.parquet` and `.faiss` artefacts and will instruct you to run the pipeline first if they’re missing.&#x20;

---

## Testing

```bash
pytest            # runs test_query.py
```

Or, ad-hoc:

```bash
python test_query.py "platform work minimum wage directive"
```

Returns a GitHub-style table with the five closest documents plus distances.&#x20;

---

## Outputs & Caching

| File/Dir                        | Producer       | Purpose                                    |
| ------------------------------- | -------------- | ------------------------------------------ |
| `horizon_scanning_combined.csv` | pipeline       | master table                               |
| `horizon_scanning_meta.parquet` | pipeline       | same, binary                               |
| `horizon_scanning.faiss`        | pipeline       | vector index                               |
| `.cache/`                       | zero-shot step | JSON memoised batches (delete to refresh)  |

---

## Extending

### Swap embedding model

```python
# in retrieval_backend.py
from sentence_transformers import SentenceTransformer
_EMBED_MODEL = SentenceTransformer("all-mpnet-base-v2")
```

Re-run the pipeline afterwards so embeddings & FAISS match.

### Adjust trend score / keywords

Open `policy_signal_scanner_v3.py` and edit:

```python
SCOPUS_QUERY = "labour mobility OR ..."
policy_keywords = yaml.safe_load(open("keywords.yaml"))["policy"]
```

### Build your own UI

`app_streamlit.py` is 260 lines—fork it or wrap the backend in FastAPI.

---

## Troubleshooting

| Symptom                                                | Fix                                                                             |
| ------------------------------------------------------ | ------------------------------------------------------------------------------- |
| `❌ Retrieval index not found – run the scraper first.` | Execute `policy_signal_scanner_v3.py` before the agent/UI.                      |
| `Scopus API error 401`                                 | Invalid / expired `SCOPUS_API_KEY`.                                             |
| CUDA out-of-memory                                     | Set `torch` to CPU or lower batch sizes in Stage 8.                             |
| Answers don’t reflect new data                         | Delete `.cache/` and rerun zero-shot step, then rebuild FAISS.                  |
| `faiss.IndexFactoryError`                              | Embedding dimension mismatch – scrape & query must use the **same** model name. |

Set `LOGLEVEL=DEBUG` for verbose logs.

---

## License

Released under the **MIT License** – see `LICENSE`.&#x20;
Feel free to open issues & PRs.

Dal Borgo R. (2025) AI-Powered Horizon-Scanning Pipeline, v0.1.*

```
