# Agentic EV Infrastructure Planning & Demand Prediction
**Newton School of Technology | Capstone Project 15**

This project implements a state-of-the-art dual-pipeline ecosystem. It merges a **High-Accuracy Machine Learning Time-Series Forecaster** with an **Autonomous AI Agentic Orchestration Layer** to seamlessly forecast EV charging stress and actively deploy intelligent grid-management strategies.

## Team Contributions
* **Krishna:** Data Pipeline Engineering & ML Model Training
* **Saumya Mishra:** Data Cleaning, Imputation & Quality Assurance
* **Rachit Gupta:** 
  - **Agentic Ecosystem Architecture**: Constructed the multi-node, recursive LangGraph orchestration state network (Reasoning, Evaluator, Planner loops).
  - **Enterprise UI/UX Build**: Engineered the professional Command Center Streamlit dashboard, interactive plotting metrics, and format-stripped RAG knowledge visualizers.
  - **API Security & Cloud Routing**: Implemented `.env` local tracking abstraction and ultra-low-latency upstream endpoint load-balancing (OpenRouter).
  - **ML Pipeline Fortification**: Programmed robust baseline time-series fallback heuristics, neutralizing missing data-index key errors natively during structural inference generation.
* **Aditya Rana:** Performance Analytics & System Evaluation

---

## 🚀 The AI Architecture Stack (Why We Used What We Used)

### 1. LangGraph (The Agentic Orchestration Engine)
Traditional LLM wrappers simply output text. We used **LangGraph** because it allows us to build a deterministic, cyclical *State Graph*. 
- Our ecosystem contains 5 autonomous AI nodes: `Reasoning Engine` -> `RAG Retriever` -> `Planner` -> `Simulator` -> `Evaluator`.
- **Why LangGraph?** If the generated plan is too vague or hallucinates, the `Evaluator` node catches it and physically routes the state payload *backwards* in a recursive loop to force the AI to self-correct.

### 2. Retrieval-Augmented Generation (RAG) via FAISS
Base AI models lack knowledge of localized infrastructure rules. Rather than fine-tuning a massive model, we utilized **RAG** (Retrieval-Augmented Generation).
- **FAISS Vector Store**: We embedded local textual files (like `ev_planning_rules.txt`) into high-dimensional vectors cleanly chunked into sections.
- **Why FAISS (Local ChromaDB Alternative)?** FAISS is globally tested for exceptionally fast similarity search without needing massive graphical overhead or external cloud databases. It cleanly grabs the exact engineering guidelines required for the AI during the active prompt phase.

### 3. OpenRouter API & Intelligent Load Balancing
We routed our LLM deployment through the **OpenRouter** ecosystem rather than locking directly into a singular API (like OpenAI natively).
- **The Tradeoff (Quality vs. Latency)**: We actively experimented with dynamic auto-routers (like `openrouter/free`) to achieve near-instant latency. However, high-speed models failed to produce structurally dense architectures and lacked deep contextual understanding.
- **The Ultimate Choice**: We intentionally hard-coded the AI to route exclusively through **`nvidia/nemotron-3-super-120b-a12b:free`**, a massive 120-billion parameter titan. While it incurs higher upstream latency, its capacity for strict JSON output adherence, multi-variable logic deduction, and uncompromised enterprise-grade reasoning outputs makes it vastly superior to lighter alternatives. We prioritized flawless algorithmic logic over sheer processing speed.

### 4. Machine Learning Backend (Random Forest)
Before the AI touches anything, we execute classical mathematical pipelines.
- **Autoregressive Features**: Extracted explicit time-delayed memory markers (`Demand_Lag_1-3`) and cyclic averages.
- **Strict Chronological Splitting**: We stripped random `.train_test_split` methods that caused data leakage, shifting into 80/20 top-down boundaries.

---

## Project Structure
* `/agent/`: The autonomous LangGraph nodes, RAG tools, and AI ecosystem.
* `/data/`: Consolidated EV station datasets for inference processing.
* `/models/`: Serialized model binaries and `.pkl` data scaling caches.
* `/src/`: Streamlit Application dashboard scripts.
* `generate_model.py`: Training engine containing the entire Time-Series ML script. 
* `.env`: Secure OpenRouter API mapping boundary.