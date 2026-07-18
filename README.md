# CoT4POI

CoT4POI is a research prototype for **Next Point-of-Interest (Next-POI) recommendation**. It explores how large language models can be used beyond direct ID prediction by adding explicit intermediate reasoning signals: temporal semantics, preference abstraction, macro-category prediction, and fine-grained category reranking.

The project was developed for an undergraduate thesis on spatio-temporal commonsense reasoning and hierarchical chain-of-thought recommendation.

## Production ML Smoke Track

This repository now includes a local production-like path for deterministic CPU validation:

```text
typed data contracts -> original-split audit -> train-only candidate index
-> B0-B3 ranking/evaluation -> local MLflow -> FastAPI -> Streamlit
-> privacy-safe monitoring replay -> CPU Docker configuration
```

The dynamically verified backend is `smoke`, a real popularity/transition/time/history ranker.
It is not a random or hard-coded mock. Legacy Fine/QLoRA code is outside the public production
tree, while its incomplete local assets are inventoried by the production compatibility manifest.
The required main weights are incomplete on this machine, so `full-gpu` is represented only by a
truthful `static_only` manifest and has no claimed runtime, latency, memory, or parity result.

![Production ML architecture](reports/figures/production_ml_architecture.drawio.png)

The editable diagram is in
[`reports/figures/production_ml_architecture.drawio`](reports/figures/production_ml_architecture.drawio).
One-page summaries are available in
[`English`](reports/production_ml_one_page.en.md) and
[`中文`](reports/production_ml_one_page.zh.md).

### Install the locked CPU environment

Python 3.10-3.12 is supported. The commands below create only a project-local environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install -e . --no-deps --no-build-isolation
.venv/bin/python -m pip check
```

### Run the synthetic closure

The tracked fixture has synthetic identities, no coordinates, and eight labeled evaluation
examples. Use a new output directory for each run because the pipeline refuses to overwrite
lineage artifacts.

```bash
.venv/bin/next-poi-evaluate \
  --dataset synthetic \
  --train tests/fixtures/synthetic/train.csv \
  --validation tests/fixtures/synthetic/validation.csv \
  --test tests/fixtures/synthetic/test.csv \
  --output artifacts/smoke-run \
  --tracking-directory mlruns \
  --split-protocol fixture-v1

.venv/bin/next-poi-serving-smoke \
  --bundle artifacts/smoke-run/b3/bundle
```

The first command runs B0-B3 plus three ablations, writes data/model/release manifests and
evaluation reports, and logs seven local MLflow runs. The second command loads the verified B3
bundle and exercises all five HTTP endpoints in process.

### Serve API, demo, and MLflow locally

```bash
NEXT_POI_BUNDLE=artifacts/smoke-run/b3/bundle \
NEXT_POI_MONITORING_PATH=monitoring_events/events.jsonl \
  .venv/bin/uvicorn next_poi.serving.app:app --host 127.0.0.1 --port 8000

NEXT_POI_API_BASE_URL=http://127.0.0.1:8000 \
  .venv/bin/streamlit run src/next_poi/demo/app.py

.venv/bin/mlflow ui --backend-store-uri ./mlruns --port 5000
```

The Streamlit app contains no model logic; it calls the FastAPI contract over HTTP.

| Endpoint | Contract |
|---|---|
| `GET /health` | Process liveness; does not claim bundle readiness |
| `GET /ready` | Verified bundle/config/index availability |
| `GET /version` | Release, data, and model versions |
| `POST /recommend` | Target-blind history + target time + Top-K request |
| `GET /metrics` | Aggregate request/status/latency counters only |

Example recommendation request:

```bash
curl -sS http://127.0.0.1:8000/recommend \
  -H 'content-type: application/json' \
  -d '{
    "dataset": "synthetic",
    "history": [{
      "poi_id": "unknown-poi",
      "category_name": "unknown-category",
      "timestamp": "2026-01-01T08:00:00Z"
    }],
    "target_time": "2026-01-01T09:00:00Z",
    "top_k": 5,
    "profile": "smoke"
  }'
```

### Monitoring replay

Monitoring JSONL contains only an explicit aggregate allowlist: version, history-length bucket,
unknown/candidate counts, hashed category buckets, source counts, entropy, latency, and status.
It never stores user ID, full history, coordinates, target, or online accuracy.

```bash
.venv/bin/next-poi-monitoring-replay \
  --reference monitoring_events/reference.jsonl \
  --current monitoring_events/current.jsonl

.venv/bin/next-poi-monitoring-replay \
  --reference monitoring_events/reference.jsonl \
  --current monitoring_events/current.jsonl \
  --inject-drift
```

The verified synthetic replay produced no alert for identical windows and reproducibly alerted
on injected category/history drift. See
[`reports/monitoring/synthetic_replay_summary.json`](reports/monitoring/synthetic_replay_summary.json).

### Use a local NYC volume/path

NYC data is ignored and is never copied into a container or report. The local reader preserves
the caller-supplied original train/validation/test split:

```bash
.venv/bin/next-poi-evaluate \
  --dataset nyc \
  --train datasets/nyc/raw/NYC_train.csv \
  --validation datasets/nyc/raw/NYC_val.csv \
  --test datasets/nyc/raw/NYC_test.csv \
  --output artifacts/nyc-smoke-run \
  --tracking-directory mlruns \
  --split-protocol original-nyc \
  --variant b3
```

### Verified results

All values below came from reruns on 2026-07-18 and are tagged `production_current`/`rerun`.
The synthetic sample is a contract fixture, so its quality values are not benchmark claims.

| Data / variant | Samples | Hit@1 | Hit@5 | Hit@10 | MRR | NDCG@10 | Candidate R@100 | Validity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Synthetic B3 | 8 | 0.375 | 1.000 | 1.000 | 0.592 | 0.692 | 1.000 | 1.000 |
| Local NYC B3 | 17,588 | 0.0706 | 0.1561 | 0.1987 | 0.1078 | 0.1294 | 0.3651 | 1.000 |

The local NYC B3 evaluator completed in 7.77 s on an Apple M4/16 GB machine with Python 3.10.11.
Its isolated in-process prediction latency was p50 0.371 ms and p95 0.418 ms; these are neither
HTTP/container latency nor GPU/online production measurements. Aggregate evidence:

- [`reports/evaluation/synthetic_smoke_matrix.md`](reports/evaluation/synthetic_smoke_matrix.md)
- [`reports/evaluation/nyc_local_smoke_summary.json`](reports/evaluation/nyc_local_smoke_summary.json)

The NYC split audit found 83,228/10,339/10,374 train/validation/test events, zero exact-event
overlap, and 7/0/10 session-identity overlaps for train-validation/train-test/validation-test.
Those findings are reported without repartitioning the frozen benchmark.

### Leakage, lineage, and compatibility boundaries

- `LabeledExample` is offline-only. `RecommendationRequest` forbids undeclared fields, including
  target/label/result, and the candidate/ranker APIs cannot accept a label.
- Known category taxonomy may span splits; POI/user encoders and all popularity, time, category,
  and transition counts are fit from train only.
- Stable sample identity is a SHA-256 of the documented six-field preimage; joins never depend on
  list position.
- Bundle load verifies schema, config, file sizes, hashes, and index invariants before readiness.
- MLflow distinguishes new reruns from archived thesis evidence with explicit lineage/source tags.
- Raw data, weights, adapters, experiment state, MLflow stores, monitoring events, and generated
  artifacts are ignored. Only the privacy-safe synthetic fixture and aggregate reports are tracked.
- The current non-loading GPU inventory is recorded in
  [`reports/evaluation/full_gpu_static_manifest_summary.json`](reports/evaluation/full_gpu_static_manifest_summary.json).

### Docker and CI

`Dockerfile.api`, `Dockerfile.demo`, and `docker-compose.yml` define CPU-only, non-root API,
Streamlit, and MLflow services with narrow build context and isolated writable volumes. Static
validation is available without Docker:

```bash
.venv/bin/python scripts/static_deploy_check.py
```

Docker Desktop is not installed in the implementation environment, so `docker compose up --build`
is deliberately `DEFERRED`, not passed. After installation:

```bash
cp .env.example .env
# Point NEXT_POI_BUNDLE_HOST_PATH at a generated B3 bundle.
docker compose config
docker compose up --build
```

The GitHub Actions workflow installs the lock, runs lint/compile/tests/static deployment checks,
then performs the synthetic B3 evaluation and API closure. Local parity commands are:

```bash
.venv/bin/ruff check src tests scripts
.venv/bin/python -m compileall -q src tests scripts
.venv/bin/python -m pytest -q
.venv/bin/python scripts/static_deploy_check.py
```

## Why This Project

Most Next-POI recommenders represent places as IDs or dense embeddings. This works for pattern fitting, but it makes user intent hard to inspect and can struggle with sparse, long-tail POIs. CoT4POI adds a lightweight reasoning pipeline before the final POI prediction:

```text
trajectory -> temporal semantics -> preference -> macro category -> fine category -> POI ID
```

The goal is not to make the LLM "think aloud" for users, but to turn implicit mobility patterns into structured signals that can be tested, ablated, and injected into the final recommendation model.

## Main Ideas

- **Temporal semantic discretization**: converts raw timestamps into human-readable temporal context such as time-of-day and day-type signals.
- **Preference abstraction**: uses an LLM to summarize recent trajectory behavior into a concise user preference description.
- **Macro-to-fine reasoning**: first predicts coarse behavioral categories, then reranks fine-grained POI categories.
- **Final LLM recommendation**: injects preference, macro, and fine signals into the final supervised fine-tuning data for POI ID prediction.
- **Ablation workspace**: keeps module-level ablations separate from the main pipeline.

## Repository Structure

```text
CoT4POI/
├── src/next_poi/                  # Production package
├── tests/                         # Unit, data, model, API, and integration checks
├── scripts/                       # Static deployment validation
├── reports/                       # Architecture and aggregate acceptance evidence
├── Dockerfile.api
├── Dockerfile.demo
├── docker-compose.yml
├── pyproject.toml
├── requirements.lock
└── README.md
```

Large local assets are intentionally not committed:

- raw and processed datasets
- model checkpoints
- API keys and `.env` files
- experiment outputs and logs
- the original thesis pipeline, retained only in the ignored local `legacy_research/` archive

## Legacy Research Archive

The original preprocessing, CoT, ablation, QLoRA, and final-evaluation sources have been moved to
the ignored local `legacy_research/` directory. They are not packaged, copied into Docker images,
checked by production CI, or included in new public checkouts. Historical Git revisions retain the
original research implementation when thesis reproduction is required.

## Acknowledgements

This project builds on ideas and implementation patterns from:

- LLM4POI: Large Language Models for Next Point-of-Interest Recommendation
- LongLoRA
- STHGCN

These works provided important references for long-context fine-tuning, POI recommendation, and spatio-temporal mobility modeling.
