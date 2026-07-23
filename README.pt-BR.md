**🌐 [English](README.md) | [Español](README.es.md) | Português**

# NLP MLOps Classifier

Um Monolito Modular de nível empresarial, pronto para produção, projetado para classificação de intenção de solicitações de suporte ao cliente de alto throughput. Este projeto utiliza Transformers de NLP com fine-tuning, padrões robustos de design de software e um pipeline de DevOps completamente automatizado.

## 📸 Screenshots

| FastAPI (Swagger) | Grafana — métricas de infra + drift |
|---|---|
| ![Swagger UI](docs/screenshots/swagger-api-docs.png) | ![Grafana dashboard](docs/screenshots/grafana-dashboard.png) |

| Rastreamento de experimentos com MLflow |
|---|
| ![MLflow](docs/screenshots/mlflow-experiments.png) |

## 🏛️ Design do sistema e decisões arquiteturais

Seguindo princípios de engenharia de sistemas de alta escala, esta plataforma rejeita o over-engineering arquitetural (como microsserviços prematuros) em favor de uma **Arquitetura Monolítica Modular** estruturada sob **Arquitetura Hexagonal (Ports & Adapters)**. `src/domain/` contém lógica de negócio livre de framework (models, ports, services) sem nenhuma dependência de FastAPI, SQLAlchemy ou PyTorch; `src/infrastructure/` contém cada adaptador concreto, conectados em uma única raiz de composição (`src/infrastructure/api/main.py`).

### Critérios técnicos principais:
* **Canal de alta leitura (Inferência):** Tokenização de texto e inferência do modelo de baixa latência, executadas fora do event loop via `run_in_threadpool` sob `torch.inference_mode()`, para que chamadas bloqueantes do PyTorch nunca travem o servidor assíncrono.
* **Canal de alta escrita (Logs de auditoria - Padrão Fan-In):** Escrever logs de transação diretamente no PostgreSQL a cada requisição HTTP criaria um gargalo no sistema. Em vez disso, um padrão **Fan-In** é implementado: `POST /predict` responde imediatamente enquanto uma `asyncio.Queue` limitada e uma única tarefa em background agrupam e persistem as entradas de auditoria.
* **Resiliência (Circuit Breaker & Timeouts):** As escritas em lote no banco de dados são limitadas a um **timeout estrito de 500ms**. Se o banco de dados sofrer uma falha temporária ou carga alta, um **Circuit Breaker** feito à mão (`CLOSED → OPEN → HALF_OPEN`) abre automaticamente. O sistema continua servindo inferências de IA com sucesso enquanto desvia os logs para um buffer de fallback em memória (`collections.deque`) até o banco de dados se recuperar.

---

## 🚦 Fases do projeto

* **Fase 1 — Benchmark local com GPU** (`train/train_phase1_benchmark.py`): fine-tuning do DistilBERT em `ag_news` (classificação de tópicos de notícias, 4 classes). Rodado de verdade em uma GTX 1650 (4GB VRAM): ~7h, **94.35% de acurácia**. Isso validou o pipeline de treinamento local com CUDA de ponta a ponta antes de investir tempo de GPU no dataset real do produto — mantido como está (comentários em espanhol) como um artefato histórico honesto.
* **Fase 2 — Modelo de produto** (`train/train_intent_classifier.py`): fine-tuning do DistilBERT no [`bitext/Bitext-customer-support-llm-chatbot-training-dataset`](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) (26.872 linhas, 27 classes de intenção balanceadas) — este é o modelo que é de fato servido atrás do `/predict`. O treinamento roda localmente em GPU pelo mantenedor; o artefato resultante é promovido para `src/infrastructure/ml_model/weights/` antes de ser empacotado na imagem Docker.

## 📦 Status atual do deploy

* A imagem Docker builda com sucesso (`docker build .`) e passa por smoke-test (boot → `/health` → `/predict`) a cada push para `main` via `.github/workflows/ci-deploy.yml`, depois é publicada no GHCR usando o `GITHUB_TOKEN` automático do repositório — nenhum secret extra é necessário para essa parte.
* `render.yaml` é um Render Blueprint completo e validado (runtime Docker, health check) — implantado e verificado funcionalmente de ponta a ponta (carregamento do modelo a partir do Hub, `/predict`, escritas de auditoria no Postgres via TLS). **Foi desativado**: o free tier do Render (512MB RAM) não comporta PyTorch + Transformers de forma confiável — medido em ~350MB apenas para o import de `torch`+`transformers`, antes de carregar o modelo — então o container acaba sendo morto por falta de memória sob carga real. Essa é uma limitação da camada de hospedagem, não um defeito de código: a mesma imagem roda bem localmente (ver "Rodar localmente") e rodaria bem no menor plano *pago* do Render sem nenhuma mudança de código. O free tier do Render também permite apenas um Postgres gerenciado por conta, então `DATABASE_URL` é uma variável de ambiente não gerenciada apontando para um Postgres externo gratuito ([Neon](https://neon.tech)) em vez de um banco gerenciado pelo Render — ver "Deploy" abaixo.
* O modelo treinado da Fase 2 nunca é commitado no git (267MB, no `.gitignore`) — em vez disso, ele fica hospedado no [Hugging Face Hub](https://huggingface.co/hard717/intent-classifier-customer-support). `MODEL_PATH` é um caminho local (`docker-compose`, promovido após um treinamento local) ou um repo id do Hub (Render), e `AutoModel.from_pretrained` lida com os dois casos de forma transparente, sem nenhuma ramificação de código.
* **O pipeline de promoção de modelo (`.github/workflows/model-promotion.yml`) é latente por design.** Ele compara o último run do MLflow com o modelo com alias `production` no MLflow Model Registry e o promove se não for pior (ver `train/promote_model.py`) — mas não faz nada até que `MLFLOW_TRACKING_URI` seja adicionado como secret do repositório, apontando para um servidor MLflow *acessível*, já que a instância local do docker-compose não é acessível a partir de um runner hospedado no GitHub. O re-treinamento automático real de ponta a ponta também precisa de um runner com GPU (o treinamento leva horas em uma GPU de consumo); essa parte permanece como uma etapa manual deliberada por enquanto.

---

## 🛠️ Stack tecnológica

* **Backend principal:** Python 3.11 + FastAPI (framework assíncrono)
* **Motor de IA:** PyTorch + Hugging Face Transformers (DistilBERT base)
* **Persistência:** PostgreSQL 16 via SQLAlchemy 2.0 (async) + migrações Alembic
* **Infraestrutura e segurança:** Nginx (Proxy Reverso, Rate-Limiting, terminação SSL)
* **DevOps e IaC:** Docker, Docker Compose, Render Blueprints (`render.yaml`)
* **Pipeline de CI/CD:** GitHub Actions — `ci-pipeline.yml` (lint + testes unitários + integração com Postgres) e `ci-deploy.yml` (build, smoke-test, publicação no GHCR)
* **MLOps — Rastreamento de experimentos:** MLflow (servidor local, backend sqlite) — cada run de treinamento da Fase 2 registra hiperparâmetros, acurácia/F1 por época, e o artefato final do modelo
* **MLOps — Monitoramento em produção:** Prometheus (latência/throughput da API, coletado de `/metrics`) + Grafana (dashboards sobre o Prometheus e diretamente sobre a tabela `audit_logs` do Postgres para métricas de modelo/negócio)

---

## 📂 Estrutura do repositório

```text
nlp-mlops-classifier/
│
├── .github/workflows/
│   ├── ci-pipeline.yml          # Lint + testes unitários + integração com Postgres (etapa CI)
│   ├── ci-deploy.yml            # Build, smoke-test, publicação no GHCR (etapa CD)
│   └── model-promotion.yml      # Latente: avaliação+promoção no MLflow, no-op até configurar MLFLOW_TRACKING_URI
│
├── src/                         # Núcleo da Arquitetura Hexagonal
│   ├── domain/                  # Lógica de negócio livre de framework: models, ports, services
│   └── infrastructure/          # Adaptadores externos (API, DB, Transformers)
│       ├── api/                 # Factory do app FastAPI, routers, schemas — raiz de composição
│       ├── database/            # Models/sessão do SQLAlchemy, repo de auditoria, batch writer Fan-In
│       ├── ml_model/            # Adaptador de inferência PyTorch + weights/ (no .gitignore)
│       ├── observability/       # Gauges do Prometheus: estado do circuit breaker, drift de confiança
│       └── resilience/          # Circuit Breaker feito à mão
│
├── alembic/                     # Ambiente de migrações async (tabela audit_logs)
│
├── train/                       # Ambiente de treinamento local isolado (GPU/CUDA)
│   ├── common.py                 # Helper compartilhado detect_device()
│   ├── train_phase1_benchmark.py # Fase 1: benchmark em ag_news com GPU (histórico, espanhol)
│   ├── train_intent_classifier.py# Fase 2: modelo de intenção de suporte ao cliente, rastreado no MLflow
│   └── promote_model.py          # Registry do MLflow: promove o último run se não regredir
│
├── infra/                       # Infraestrutura como Código (IaC)
│   ├── nginx/                   # Perfis de roteamento do Proxy Reverso
│   ├── prometheus/              # Config de scraping para o /metrics da API
│   ├── grafana/                 # Provisionamento de datasources + dashboard (Prometheus + Postgres)
│   └── docker-compose.yml       # Orquestração local de 6 containers (API, DB, proxy + stack de MLOps)
│
├── docs/screenshots/            # Screenshots do README (Swagger, Grafana, MLflow)
├── render.yaml                  # Manifesto do Render Blueprint (raiz do repo — local padrão do Render)
├── tests/                       # Suite de Pytest (unitários + @pytest.mark.integration)
├── Dockerfile                   # Build multi-stage de produção (torch CPU-only)
├── requirements.txt             # Dependências de produção
├── requirements-dev.txt         # + pytest, httpx, flake8
└── LICENSE                      # Licença MIT
```

---

## 🚀 Rodar localmente

```bash
cd infra
docker-compose up --build
```

Isso sobe seis containers: `postgres_db`, `api_service`, `nginx_proxy`, `mlflow`, `prometheus` e `grafana`. Uma vez saudáveis:

```bash
curl http://localhost/health
curl -X POST http://localhost/predict -H "Content-Type: application/json" \
  -d '{"text": "I want to cancel my order"}'
```

Requer um artefato de modelo promovido em `src/infrastructure/ml_model/weights/` (ver "Retreinar" abaixo) — o container `api_service` vai falhar ao iniciar sem ele.

## 📈 Monitoramento e observabilidade

* **MLflow** (`http://localhost:5000`): cada run de treinamento da Fase 2 aparece aqui automaticamente — hiperparâmetros, acurácia/F1 por época, e o artefato do modelo salvo. Suba-o antes de treinar: `docker compose -f infra/docker-compose.yml up -d mlflow`. `train/train_intent_classifier.py` aponta para `http://localhost:5000` por padrão (sobrescreva com `MLFLOW_TRACKING_URI`).
* **Prometheus** (`http://localhost:9090`): coleta `GET /metrics` da API em execução a cada 5s — taxa de requisições, histogramas de latência por rota, e o estado atual do circuit breaker do banco de auditoria (`circuit_breaker_state`: 0=fechado, 1=aberto, 2=half_open).
* **Grafana** (`http://localhost:3000`, acesso anônimo de visualização habilitado — `admin`/`admin` para editar): o dashboard *"NLP MLOps Classifier - Overview"* é provisionado automaticamente na inicialização com dois tipos de painéis — métricas de infra do Prometheus (taxa de requisições, latência p95, estado do breaker) e métricas de produto consultadas diretamente da tabela `audit_logs` (distribuição de intents, confiança média ao longo do tempo, volume de predições).
* **Detecção de drift** (`src/infrastructure/observability/drift.py`): uma tarefa em background dentro da API recalcula a confiança média móvel das predições a cada `drift_check_interval_seconds` (5 min por padrão) e a compara com `drift_baseline_confidence` (a confiança do conjunto de validação da Fase 2). A diferença é exposta como `prediction_drift_score` em `/metrics` — visível no Grafana — e registrada como warning ao ultrapassar `drift_alert_threshold`. É um proxy sem labels: o drift real de acurácia precisa de labels verdadeiros que o tráfego de produção não tem, mas uma queda sustentada na confiança costuma ser o primeiro sintoma visível.

## 🔁 Retreinar

```bash
pip install -r requirements-dev.txt
docker compose -f infra/docker-compose.yml up -d mlflow   # opcional mas recomendado: habilita o tracking
python -m train.train_intent_classifier
# promover o artefato escolhido:
cp -r models/intent_classifier_customer_support/* src/infrastructure/ml_model/weights/
# e, para servi-lo a partir de um repo que nunca envia os pesos (ex. Render):
hf upload <seu-usuario-hf>/intent-classifier-customer-support models/intent_classifier_customer_support .
```

## ☁️ Deploy (Render + Neon + Hugging Face Hub)

O free tier de cada peça aqui tem uma pegadinha, então as peças são separadas em vez de usar o banco de dados tudo-em-um do Blueprint do Render:

1. **Modelo**: envie o artefato promovido para um repo de modelo no Hugging Face Hub (ver "Retreinar" acima). Configure `MODEL_PATH` com esse repo id em vez de um caminho local.
2. **Banco de dados**: o plano gratuito do Render permite apenas um Postgres gratuito ativo por conta. Crie um projeto gratuito no [Neon](https://neon.tech) em vez disso, e copie sua connection string.
3. **Web service**: [dashboard.render.com/blueprint/new](https://dashboard.render.com/blueprint/new) → aponte para este repo/branch → o Render lê o `render.yaml` e provisiona o web service `nlp-mlops-classifier` (sem banco de dados, já que o `render.yaml` não define mais um).
4. No serviço criado → **Environment**, configure `DATABASE_URL` com a connection string do Neon (deixada como variável não gerenciada/manual no blueprint de propósito).
5. Opcional: adicione `RENDER_DEPLOY_HOOK_URL` (Serviço → Settings → Deploy Hook) como secret do GitHub Actions para ativar o passo de auto-deploy já presente no `ci-deploy.yml`.

## ✅ Testes

```bash
pip install -r requirements-dev.txt
flake8 .
pytest tests/ -m "not integration"      # não precisa de serviços externos
pytest tests/ -m integration            # requer Postgres, ex. `docker-compose up postgres_db`
```
