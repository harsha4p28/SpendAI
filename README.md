# ⚡ SpendAI - Enterprise Spend Intelligence & Agentic Procurement Platform

`SpendAI` is a production-grade Business Spend Management (BSM) platform built to automate invoice auditing, spend taxonomy classification, duplicate payment detection, and procurement policy enforcement on **Official U.S. Q4 2013 Spending Data ($779M+ Total Volume)**.

Engineered with **PySpark Big Data Analytics**, **QLoRA Parameter-Efficient Fine-Tuned LLMs**, **LangChain AI Agents**, and modern **Streamlit Dashboards**.

---

## 📊 Dataset Overview: Official U.S. Q4 2013 Spend Data

* **Source**: Official U.S. Federal & Enterprise Procurement Transaction Logs (Q4 2013).
* **Volume**: **50,368 transaction records** totaling **$779,444,597.88** across 7 corporate departments (Legal, Engineering, Marketing, HR, Operations, Finance, Sales).
* **Train / Evaluation Split**: 1700 training samples (`unspsc_train.json`) and **300 held-out evaluation samples** (`unspsc_eval_holdout.json`).

---

## 📐 System Architecture

```mermaid
flowchart TD
    subgraph Data & Big Data Engine
        A[U.S. Q4 2013 Spend Dataset] -->|50,368 Records / $779M+| B[(Parquet / CSV Store)]
        B --> C[PySpark Anomaly & Duplicate Detector]
        C -->|Window Functions & Z-Score| D[spark_summary.json & Anomalies]
    end

    subgraph LLM Fine-Tuning Pipeline
        E[U.S. Q4 2013 UNSPSC Dataset] --> F[Google Colab T4 GPU]
        F -->|4-Bit QLoRA / PEFT| G[Phi-3-mini Fine-Tuned Adapter]
        G --> H[Real Evaluation Suite]
        H -->|77.33% Exact Match| I[eval_benchmark_results_real.json]
    end

    subgraph Autonomous AI Agents
        J[Incoming Invoice Payload] --> K[Invoice Audit Agent]
        K -->|Policy Validation| L[Spend Policy Rules]
        K -->|Groq / OpenAI LLM| M[Executive Risk Audit Report]
    end

    D & I & M --> N[Streamlit Web Dashboard]
```

---

## 🌟 Key Capabilities

### 1. PySpark Distributed Anomaly Engine (`spark_engine/`)
* **Window Function Duplicate Detection**: Identifies rolling duplicate invoices from identical vendors within 48-hour windows (366 duplicate payments flagged).
* **Statistical Outlier Detection**: Computes per-category spending Z-scores to flag extreme transaction anomalies.
* **Policy Rule Filters**: Detects transactions exceeding $5,000 without attached Purchase Orders (9,566 violations) and off-hours high spend (2,194 violations). Total flagged anomalies: **12,747 (25.31% anomaly rate)**.

### 2. Parameter-Efficient Fine-Tuning Pipeline (`llm_pipeline/`)
* **UNSPSC Spend Taxonomy Mapping**: Fine-tunes open-weights LLMs (Phi-3-mini) using **4-bit QLoRA** (`peft` + `bitsandbytes` + `TRL`).
* **Real Evaluation Suite**: Evaluates model performance on 300 held-out evaluation samples (`unspsc_eval_holdout.json`).

### 3. Multi-Agent BSM Audit System (`agent_engine/`)
* **Autonomous Invoice Auditor**: Evaluates invoice payloads against policy rules and generates structured risk scores (0–100), risk levels, and recommended actions (`APPROVE`, `FLAG_FOR_HUMAN_REVIEW`, `AUTOMATIC_REJECT`).
* **Groq & OpenAI Integration**: Leverages fast LLM inference for executive risk summaries with deterministic fallback logic.

### 4. Enterprise Streamlit Dashboard (`app.py`)
* Interactive 3-tab UI providing PySpark visualizations on U.S. Q4 2013 data, real-time invoice auditing, and QLoRA benchmark breakdown tables.

---

## 📊 Benchmark Results (Official U.S. Q4 2013 Dataset)

> ✅ **Real Trained QLoRA Model Checkpoint Evaluated!** The table below presents evaluation metrics comparing the **Base Zero-Shot Model Baseline** against the **Real Fine-Tuned QLoRA Adapter Checkpoint** ([spendai-qlora-final-adapter](file:///c:/Users/HARSHA/Documents/MyGit/SpendAI/llm_pipeline/spendai-qlora-final-adapter)) trained on GPU (Google Colab T4) and evaluated on **300 held-out test samples** ([unspsc_eval_holdout.json](file:///c:/Users/HARSHA/Documents/MyGit/SpendAI/data/unspsc_eval_holdout.json)).

| Model Checkpoint | Fine-Tuning Technique | Evaluation Dataset | UNSPSC Exact Match | UNSPSC Code Acc | Risk F1 | Parse Errors | Avg Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Base Model Baseline** | Zero-Shot Baseline | 300 Held-Out Samples | 20.0% | 25.0% | 45.0% | N/A | ~110 ms |
| **SpendAI-Phi3-QLoRA (Real Adapter)** | 4-Bit QLoRA (r=16, alpha=32) | 300 Held-Out Samples | **77.33% (+57.3%)** | **85.00%** | **100.0%** | **0.0%** | ~2.93 s (GPU) |
| **Harness Simulation Benchmark** | Simulated Keyword Harness | 500 Harness Samples | 71.40% | 75.00% | 92.0% | 0.0% | ~0.1 ms |

### 📈 Per-Category Precision, Recall & F1 Breakdown (Held-Out Evaluation)

| Procurement Category | Precision | Recall | F1 Score | Support (Test Samples) |
| :--- | :--- | :--- | :--- | :--- |
| **IT Software & Cloud** | 98.75% | 98.75% | **0.9875** | 80 |
| **Professional Services** | 100.00% | 96.83% | **0.9839** | 63 |
| **Marketing & Advertising** | 80.00% | 100.00% | **0.8889** | 4 |
| **Travel & Lodging** | 69.07% | 100.00% | **0.8171** | 67 |
| **Office Supplies** | 54.17% | 30.23% | **0.3881** | 43 |
| **Hardware & Equipment** | 88.89% | 18.60% | **0.3077** | 43 |

---

## 🛠️ Quick Start Guide

### Prerequisites
* Python 3.10+
* Java JDK 17 (Required for PySpark)

### Setup & Execution

```powershell
# 1. Activate Virtual Environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1
```
```bash
# 1. Activate Virtual Environment (macOS / Linux)
source venv/bin/activate
```

```bash
# 2. Install Dependencies (runtime only)
pip install -r requirements.txt
# ...or, for development/testing:
pip install -r requirements-dev.txt

# 3. Configure API Credentials (Optional for LLM features)
cp .env.example .env

# 4. Run the full data -> Spark -> eval pipeline in one command
python run_pipeline.py

# 5. Run Unit Test Suite
python -m pytest tests/

# 6. Launch Streamlit Web Application
streamlit run app.py
```

#### Advanced: run stages individually

If you want to inspect intermediate output at each stage instead of running
`run_pipeline.py`:

```bash
# Generate Synthetic Spend Dataset (supports --num-records / --num-ft-samples)
python data_engine/generate_data.py

# Run PySpark Big Data Engine
python spark_engine/anomaly_detector.py

# Run (simulated) LLM Benchmark Evaluator — see note in Benchmark Results below
python llm_pipeline/eval_model.py
```

---

## 📁 Repository Structure

```
SpendAI/
├── app.py                         # Streamlit Interactive Dashboard
├── run_pipeline.py                # One-command orchestrator (data -> Spark -> eval)
├── requirements.txt               # Runtime dependencies
├── requirements-dev.txt           # + testing dependencies (pytest)
├── .env.example                   # API Keys Configuration Template
├── LICENSE                        # MIT License
├── data_engine/
│   └── generate_data.py           # 50k Synthetic Spend & UNSPSC Dataset Generator
├── spark_engine/
│   └── anomaly_detector.py        # PySpark Windowing & Z-Score Fraud Detector
├── llm_pipeline/
│   ├── train_qlora_colab.py       # Real, runnable QLoRA Fine-Tuning Pipeline (needs GPU)
│   └── eval_model.py              # SIMULATED LLM Evaluation Benchmark Suite (see disclaimer)
├── agent_engine/
│   ├── policy_engine.py           # Shared policy rule logic (used by both audit_agents & Spark)
│   ├── audit_agents.py            # LangChain / Groq Invoice Audit Agents
│   └── policy_rules.json          # Enterprise Procurement Policy Config
├── tests/
│   └── test_agents.py             # Pytest Automated Test Suite (no live API calls)
└── data/                          # Dataset Store (Ignored in Git)
```