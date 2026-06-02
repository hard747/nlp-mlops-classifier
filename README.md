# NLP MLOps Classifier

An enterprise-grade, production-ready Modular Monolith designed for high-throughput customer support intent classification. This project leverages fine-tuned NLP Transformers, robust software design patterns, and a complete automated DevOps pipeline.

## 🏛️ System Design & Architectural Decisions

Following high-scale system engineering principles, this platform rejects architectural overengineering (such as premature microservices) in favor of a **Modular Monolithic Architecture** structured under **Hexagonal Architecture (Ports & Adapters)**.

### Key Technical Criteria:
* **High-Read Channel (Inference):** Synchronous, low-latency text tokenization and model inference processed entirely in-memory ($0\mu s$ network overhead between the API and the PyTorch engine).
* **High-Write Channel (Auditory Logs - Fan-In Pattern):** Writing transaction logs directly to PostgreSQL on every HTTP request would bottleneck the system. Instead, a **Fan-In** pattern is implemented: the API responds immediately to the user while delegating asynchronous log persistence to a background worker pool in structured batches.
* **Resiliency (Circuit Breaker & Timeouts):** Database interactions are constrained to a strict **500ms timeout**. If the database experiences a temporary failure or heavy load, a **Circuit Breaker** opens automatically. The system continues to serve AI inferences successfully while diverting logs to an in-memory fallback buffer until the database recovers.

---

## 🛠️ Tech Stack

* **Core Backend:** Python 3.11 + FastAPI (Asynchronous framework)
* **AI Engine:** PyTorch + Hugging Face Transformers (DistilBERT base)
* **Infrastructure & Security:** Nginx (Reverse Proxy, Rate-Limiting, SSL termination)
* **Data Persistence:** PostgreSQL 16
* **DevOps & IaC:** Docker, Docker Compose, Render Blueprints (`render.yaml`)
* **CI/CD Pipeline:** GitHub Actions (`pytest` automated quality gates)

---

## 📂 Repository Structure

```text
nlp-mlops-classifier/
│
├── .github/workflows/
│   ├── ci-pipeline.yml          # Automated testing suite (CI Stage)
│   └── cd-deploy.yml            # Automated cloud deployment (CD Stage)
│
├── src/                         # Hexagonal Architecture Core
│   ├── domain/                  # Core Business Logic & Models
│   └── infrastructure/          # External Adapters (API, DB, Transformers)
│       ├── api/                 # FastAPI HTTP Routing layer
│       ├── database/            # PostgreSQL connections & Circuit Breaker
│       └── ml_model/            # In-memory PyTorch Inference Engine
│
├── train/                       # Isolated Local Training Environment (GPU/CUDA)
│   ├── dataset/                 # Customer support training data (.csv)
│   └── train.py                 # Fine-tuning script optimized for NVIDIA CUDA
│
├── infra/                       # Infrastructure as Code (IaC)
│   ├── nginx/                   # Reverse Proxy routing profiles
│   ├── docker-compose.yml       # 3-Container Local Orchestration
│   └── render.yaml              # Production Render Blueprint Manifest
│
├── tests/                       # Automated Pytest Suite
├── Dockerfile                   # Multi-Stage Production Build
├── requirements.txt             # Production dependencies
└── LICENSE                      # MIT License