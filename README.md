````markdown
# Policy Horizon Scanner 🛰️

*A Python toolkit that periodically (or on-demand) harvests labour-market & social-policy signals from open sources, enriches them with NLP, stores semantic embeddings in FAISS, and answers questions through a Retrieval-Augmented Generation (RAG) agent (CLI **and** Streamlit UI).*

*Although pre-configured for labour-market and social-policy content, the pipeline is domain-agnostic — drop in a new **profile** file (feeds + keywords) and you’re ready for climate tech, healthcare innovation, or any other topic.*

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

| # | Stage | What happens | Main libs |
|---|-------|--------------|-----------|
| **1** | Harvest | Async-fetch profile feeds + Scopus + OpenAIRE | `httpx`, `feedparser`, `tenacity` |
| **2** | Clean + Filter | De-dupe, robust date parse, **policy keyword** filter | `pandas`, `re`, `PyYAML` |
| **3** | Enrich | Trend score, TF-IDF tags, **PESTLE** zero-shot, **stakeholder NER** | `transformers`, `spaCy`, `disk_cache` |
| **4** | Weak-signal flag | `keyword_freq` → `rarity_score` (≤ 2 cues ⇒ rare) | `pandas` |
| **5** | Embed + Index | MiniLM embeddings → **FAISS** | `sentence_transformers`, `faiss-cpu` |
| **6** | Review | Top trend / rare items → `review_items.parquet`; Streamlit tab shows ✓/✗ queue | `pandas`, `streamlit` |
| **7** | Retrieve | `retrieval_backend.snippets()` → k-nearest docs | `faiss`, `numpy` |
| **8** | Answer | `agent_app.py` builds prompt & queries **GPT-4o** | `openai` |
| **9** | Visualise | Streamlit dashboard: KPIs, Weak-Signals tab, chat, review | `streamlit` |
| **10** | Test | Minimal `pytest` suite in `/tests` | `pytest`, `tabulate` |

---

## Folder Layout

```text
.
├── README.md  ·  LICENSE  ·  requirements.txt  ·  .gitignore
│
├── config/
│   ├── config.sample.yaml        # dummy SCOPUS key, etc.
│   └── config2.sample.json       # dummy OpenAI key
│
├── profiles/
│   └── global.yml                # feeds + keywords
│
├── src/
│   └── horizon_scanner/
│       ├── __init__.py
│       ├── embed_loader.py
│       ├── features.py
│       ├── fetchers.py
│       ├── stakeholder_ner.py
│       ├── faiss_search.py
│       ├── review_utils.py
│       └── disk_cache.py
│
├── scripts/
│   ├── policy_signal_scanner_v3.py    # main ETL / embedding pipeline
│   ├── retrieval_backend.py           # RAG helper
│   └── agent_app.py                   # CLI chat loop
│
├── streamlit/
│   ├── app_streamlit.py               # **full-featured dashboard** – multi-tab UI: KPIs, chat, review, weak signals
│   └── dashboard.py                   # **lite dashboard** – single-page KPI/quick-peek view
│
├── tools/
│   ├── inspect_parquet.py
│   └── peek.py
│
└── tests/
    ├── sanity_check.py
    ├── test_query.py
    └── test_utils.py
````

Large artefacts (`*.faiss`, `*.parquet`, combined CSV) are **git-ignored**.

---

## Quick Start

```bash
# 1 · clone & enter
git clone https://github.com/Paralogosai/ai-horizon-scanning-prototype.git
cd ai-horizon-scanning-prototype

# 2 · create env & install
python -m venv .venv && source .venv/bin/activate      # PS> .\.venv\Scripts\activate
pip install -r requirements.txt

# 3 · provide secrets (local only)
cp config/config.sample.yaml  config/config.yaml       # edit with your SCOPUS key
cp config/config2.sample.json config/config2.json      # edit with your OPENAI key

# 4 · run the pipeline  (~4-5 min on CPU)
python scripts/policy_signal_scanner_v3.py --profile global

# 5 · chat with the agent
python scripts/agent_app.py
```

---

## Configuration

| Item                            | File                                                 | Notes                               |
| ------------------------------- | ---------------------------------------------------- | ----------------------------------- |
| **Profiles (feeds + keywords)** | `profiles/*.yml`                                     | `global.yml` ships with 40 + feeds. |
| **OpenAI key**                  | `config/config2.json` → `{"OPENAI_API_KEY": "sk-…"}` | required for agent & Streamlit      |
| **Scopus key**                  | `config/config.yaml` → `SCOPUS_API_KEY: "…" `        | optional but recommended            |
| **Pipeline flags**              | top of `scripts/policy_signal_scanner_v3.py`         | e.g. `ENABLE_SUMMARY = True`        |
| **Cache TTL**                   | call in `disk_cache.clean_cache()`                   | change `max_age_days` value         |

---

## Running the Pipeline

```bash
python scripts/policy_signal_scanner_v3.py --profile global      # full run
python scripts/policy_signal_scanner_v3.py --profile demo        # tiny test profile
```

Outputs land in repo root (`*.csv`, `*.parquet`, `*.faiss`).

---

## Chatting with the Agent

```bash
python scripts/agent_app.py
```

Enter a question → retrieval\_backend pulls context → GPT-4o answers.
Type `quit` or `q` to exit.

---

## Streamlit Dashboard

```bash
streamlit run streamlit/app_streamlit.py
```

*Overview | Ask GPT-4o | Analyst Review | Weak Signals* tabs appear.
If data/index files are missing, the app prompts you to run the scraper first.

---

## Testing

```bash
pytest -q
```

Quick retrieval smoke-test:

```bash
python tests/test_query.py "platform work minimum wage directive"
```

---

## Outputs & Caching

| Artefact                        | Producer     | Purpose                |
| ------------------------------- | ------------ | ---------------------- |
| `horizon_scanning_combined.csv` | pipeline     | master table           |
| `horizon_scanning_meta.parquet` | pipeline     | metadata + flags       |
| `horizon_scanning.faiss`        | pipeline     | vector index           |
| `.cache/`                       | enrich stage | auto-pruned (> 7 days) |
| `review_items.parquet`          | pipeline     | moderation queue       |

---

## Extending

### Swap embedding model

Change `EMBED_MODEL_NAME` in `src/horizon_scanner/embed_loader.py`, then rerun the pipeline.

### Add a new profile

Create `profiles/your_topic.yml` with `feeds:` + `keywords:` and run:

```bash
python scripts/policy_signal_scanner_v3.py --profile your_topic
```

### Expose as an API

Wrap `retrieval_backend` in FastAPI or fork the Streamlit app.

---

## Troubleshooting

| Symptom                         | Fix                                                           |
| ------------------------------- | ------------------------------------------------------------- |
| **“Retrieval index not found”** | Run the scraper first.                                        |
| **Scopus 401**                  | Invalid/expired `SCOPUS_API_KEY`.                             |
| **CUDA OOM**                    | Set `TORCH_FORCE_GPU=0` or lower batch size.                  |
| **faiss.IndexFactoryError**     | Embed model changed → rebuild index.                          |
| **Stale answers**               | `rm -r .cache && python scripts/policy_signal_scanner_v3.py`. |

Set `LOGLEVEL=DEBUG` for verbose logs.

---

## License

Released under the **MIT License** – see `LICENSE`.
Feel free to open issues & PRs.

Dal Borgo, R. (2025) AI-Powered Horizon-Scanning Pipeline, v0.1.*
