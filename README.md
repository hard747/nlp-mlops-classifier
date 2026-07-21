# NLP MLOps Classifier

An enterprise-grade, production-ready Modular Monolith designed for high-throughput customer support intent classification. This project leverages fine-tuned NLP Transformers, robust software design patterns, and a complete automated DevOps pipeline.

## 🏛️ System Design & Architectural Decisions

Following high-scale system engineering principles, this platform rejects architectural overengineering (such as premature microservices) in favor of a **Modular Monolithic Architecture** structured under **Hexagonal Architecture (Ports & Adapters)**. `src/domain/` contains framework-free business logic (models, ports, services) with zero dependencies on FastAPI, SQLAlchemy, or PyTorch; `src/infrastructure/` holds every concrete adapter, wired together at a single composition root (`src/infrastructure/api/main.py`).

### Key Technical Criteria:
* **High-Read Channel (Inference):** Low-latency text tokenization and model inference, run off the event loop via `run_in_threadpool` under `torch.inference_mode()` so blocking PyTorch calls never stall the async server.
* **High-Write Channel (Auditory Logs - Fan-In Pattern):** Writing transaction logs directly to PostgreSQL on every HTTP request would bottleneck the system. Instead, a **Fan-In** pattern is implemented: `POST /predict` responds immediately while a bounded `asyncio.Queue` and a single background task batch and persist audit entries.
* **Resiliency (Circuit Breaker & Timeouts):** Database batch writes are constrained to a strict **500ms timeout**. If the database experiences a temporary failure or heavy load, a hand-rolled **Circuit Breaker** (`CLOSED → OPEN → HALF_OPEN`) opens automatically. The system continues to serve AI inferences successfully while diverting logs to an in-memory fallback buffer (`collections.deque`) until the database recovers.

---

## 🚦 Project Phases

* **Phase 1 — Local GPU benchmark** (`train/train_phase1_benchmark.py`): fine-tunes DistilBERT on `ag_news` (4-class news topic classification). Run for real on a GTX 1650 (4GB VRAM): ~7h, **94.35% accuracy**. This validated the local CUDA training pipeline end to end before committing GPU time to the actual product dataset — kept as-is (Spanish comments) as an honest historical artifact.
* **Phase 2 — Product model** (`train/train_intent_classifier.py`): fine-tunes DistilBERT on [`bitext/Bitext-customer-support-llm-chatbot-training-dataset`](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) (26,872 rows, 27 balanced intent classes) — this is the model actually served behind `/predict`. Training is run locally on GPU by the maintainer; the resulting artifact is promoted into `src/infrastructure/ml_model/weights/` before it ships in the Docker image.

## 📦 Current Deployment Status

* Docker image builds successfully (`docker build .`) and is smoke-tested (boot → `/health` → `/predict`) on every push to `main` via `.github/workflows/ci-deploy.yml`, then published to GHCR using the repo's automatic `GITHUB_TOKEN` — no extra secrets required for that part.
* `render.yaml` is a complete, validated Render Blueprint (Docker runtime, managed free-tier Postgres, health check). **Live Render activation is an intentional manual step** — the deploy step in `ci-deploy.yml` is gated behind an optional `RENDER_DEPLOY_HOOK_URL` secret and simply does nothing until that secret is added.
* The Phase 2 model artifact ships only after a local GPU training run; until then `src/infrastructure/ml_model/weights/` holds a placeholder (`.gitkeep`) and the API's model-loading step will fail to start, by design — this is not meant to run in production before that promotion step.

---

## 🛠️ Tech Stack

* **Core Backend:** Python 3.11 + FastAPI (Asynchronous framework)
* **AI Engine:** PyTorch + Hugging Face Transformers (DistilBERT base)
* **Persistence:** PostgreSQL 16 via SQLAlchemy 2.0 (async) + Alembic migrations
* **Infrastructure & Security:** Nginx (Reverse Proxy, Rate-Limiting, SSL termination)
* **DevOps & IaC:** Docker, Docker Compose, Render Blueprints (`render.yaml`)
* **CI/CD Pipeline:** GitHub Actions — `ci-pipeline.yml` (lint + unit + Postgres-integration tests) and `ci-deploy.yml` (build, smoke-test, GHCR publish)
* **MLOps — Experiment Tracking:** MLflow (local server, sqlite backend) — every Phase 2 training run logs hyperparameters, per-epoch accuracy/F1, and the final model artifact
* **MLOps — Production Monitoring:** Prometheus (API latency/throughput scraped from `/metrics`) + Grafana (dashboards over Prometheus and directly over the `audit_logs` Postgres table for model/business metrics)

---

## 📂 Repository Structure

```text
nlp-mlops-classifier/
│
├── .github/workflows/
│   ├── ci-pipeline.yml          # Lint + unit + Postgres-integration tests (CI Stage)
│   └── ci-deploy.yml            # Build, smoke-test, GHCR publish (CD Stage)
│
├── src/                         # Hexagonal Architecture Core
│   ├── domain/                  # Framework-free business logic: models, ports, services
│   └── infrastructure/          # External Adapters (API, DB, Transformers)
│       ├── api/                 # FastAPI app factory, routers, schemas — composition root
│       ├── database/            # SQLAlchemy models/session, audit repo, Fan-In batch writer
│       ├── ml_model/            # PyTorch inference adapter + weights/ (gitignored)
│       ├── observability/       # Prometheus circuit-breaker-state gauge
│       └── resilience/          # Hand-rolled Circuit Breaker
│
├── alembic/                     # Async migration environment (audit_logs table)
│
├── train/                       # Isolated Local Training Environment (GPU/CUDA)
│   ├── common.py                 # Shared detect_device() helper
│   ├── train_phase1_benchmark.py # Phase 1: ag_news GPU benchmark (historical, Spanish)
│   └── train_intent_classifier.py# Phase 2: customer-support intent model, MLflow-tracked
│
├── infra/                       # Infrastructure as Code (IaC)
│   ├── nginx/                   # Reverse Proxy routing profiles
│   ├── prometheus/              # Scrape config for the API's /metrics
│   ├── grafana/                 # Datasource + dashboard provisioning (Prometheus + Postgres)
│   └── docker-compose.yml       # 6-Container Local Orchestration (API, DB, proxy + MLOps stack)
│
├── render.yaml                  # Render Blueprint Manifest (repo root — Render's default lookup)
├── tests/                       # Pytest suite (unit + @pytest.mark.integration)
├── Dockerfile                   # Multi-Stage Production Build (CPU-only torch)
├── requirements.txt             # Production dependencies
├── requirements-dev.txt         # + pytest, httpx, flake8
└── LICENSE                      # MIT License
```

---

## 🚀 Run locally

```bash
cd infra
docker-compose up --build
```

This boots six containers: `postgres_db`, `api_service`, `nginx_proxy`, `mlflow`, `prometheus`, and `grafana`. Once healthy:

```bash
curl http://localhost/health
curl -X POST http://localhost/predict -H "Content-Type: application/json" \
  -d '{"text": "I want to cancel my order"}'
```

Requires a promoted model artifact at `src/infrastructure/ml_model/weights/` (see "Retrain" below) — the `api_service` container will fail to start without it.

## 📈 Monitoring & Observability

* **MLflow** (`http://localhost:5000`): every Phase 2 training run appears here automatically — hyperparameters, per-epoch accuracy/F1, and the saved model artifact. Start it before training: `docker compose -f infra/docker-compose.yml up -d mlflow`. `train/train_intent_classifier.py` points at `http://localhost:5000` by default (override with `MLFLOW_TRACKING_URI`).
* **Prometheus** (`http://localhost:9090`): scrapes `GET /metrics` on the running API every 5s — request rate, latency histograms per route, and the audit-DB circuit breaker's current state (`circuit_breaker_state`: 0=closed, 1=open, 2=half_open).
* **Grafana** (`http://localhost:3000`, anonymous viewer access enabled — `admin`/`admin` for editing): the *"NLP MLOps Classifier - Overview"* dashboard is provisioned automatically on startup with two kinds of panels — infra metrics from Prometheus (request rate, p95 latency, breaker state) and product metrics queried directly from the `audit_logs` table (intent distribution, average confidence over time, prediction volume).

## 🔁 Retrain

```bash
pip install -r requirements-dev.txt
docker compose -f infra/docker-compose.yml up -d mlflow   # optional but recommended: enables tracking
python -m train.train_intent_classifier
# promote the chosen artifact:
cp -r models/intent_classifier_customer_support/* src/infrastructure/ml_model/weights/
```

## ✅ Tests

```bash
pip install -r requirements-dev.txt
flake8 .
pytest tests/ -m "not integration"      # no external services needed
pytest tests/ -m integration            # requires Postgres, e.g. `docker-compose up postgres_db`
```
