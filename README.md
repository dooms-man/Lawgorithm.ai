# ⚖️ Lawgorithm.ai – AI-Driven Regulatory Compliance Automation

> **"Transforming complex regulations into actionable, explainable insights."**

---

## 🚀 Overview

**Lawgorithm.ai** is an AI-powered platform that automates regulatory compliance checks across multi-jurisdictional laws such as **GDPR**, **AMLD**, and **Basel III**.  
It parses legal documents, identifies compliance gaps, provides explainable recommendations, and maintains an auditable trail for transparency.

The system supports **department-specific dashboards** — Legal, Product, and Operations — offering a unified compliance cockpit.

---

## 🧠 Key Features

### 1. Regulatory Document Parsing
- Extracts text from PDF regulations and contracts.  
- Uses **semantic embeddings** to compare regulatory clauses with internal policies.

### 2. Compliance Gap Detection
- Identifies mismatched or missing clauses.  
- Categorizes gaps by severity (Low / Medium / High).  
- Uses hybrid **LLM + Rule-Based Comparators** for accuracy.

### 3. Actionable Recommendations & Explainability
- Generates step-by-step remediation guidance using LLMs.  
- Includes explainability via evidence sentences, similarity scores, and source clause references.

### 4. Audit Trail & Approval Workflow
- Tracks all compliance flags and user decisions.  
- Maintains immutable logs of approvals/rejections.  
- Ensures end-to-end traceability for legal audits.

### 5. Multi-Jurisdiction Awareness
- Tags every regulation and contract by jurisdiction (EU, US, APAC, etc.).  
- Automatically prioritizes **local > international > company policy** rules.  

### 6. Feedback Loop & Model Governance
- Stores user feedback (accept/reject/correct).  
- Periodically retrains classifiers (obligation, compliance area) for continuous improvement.

---

## 🧩 System Architecture

Documents (Regulations / Contracts)
│
▼
[Document Parser + Preprocessor]
│
▼
[Embedding Model + Similarity Engine]
│
▼
[Compliance Gap Detector] ─────┐
│ │
▼ ▼
[LLM Explainability Engine] [Rule-Based Comparator]
│
▼
[Suggestion Generator + Actionable Steps]
│
▼
[Audit Trail + Feedback Store + PostgreSQL DB]
│
▼
[Role-Based Dashboards: Legal | Product | Ops]


## 🧰 Tech Stack

| Layer              | Technology                               | Purpose                                |
| ------------------ | ---------------------------------------- | -------------------------------------- |
| Backend            | FastAPI, Python                          | Core API and orchestration             |
| NLP/ML             | Sentence Transformers, zephyr, spaCy     | Embedding & semantic analysis          |
| Database           | PostgreSQL                               | Compliance flags, audit logs, feedback |
| Explainability     | LLM-based explanations                   | Model transparency                     |
| Deployment         | Docker, Uvicorn                          | Containerized service                  |
| Frontend (Planned) | Vue.js / React (JSX)                     | Department dashboards                  |


## 👥 Contributors
Team: Runtime Raita
-Akshara Gupta
-Manas Shewale
-Hardik Bhalekar


