# CoT4POI to Production ML: Deterministic Next-POI Smoke System

## Objective

This refactor turns a research-oriented Next-POI repository into a reproducible, testable,
serviceable, and observable single-machine ML system. The only dynamically accepted profile is a
deterministic CPU smoke backend. Incomplete legacy Fine/QLoRA assets are inventoried through a
truthful static manifest; no full-GPU result is implemented or claimed.

## System

The offline path reads explicit train/validation/test files without repartitioning them, normalizes
events, audits split overlap and time boundaries, creates stable SHA-256 sample identities, and
fits every learned statistic on train only. A versioned candidate index supports global, temporal,
first-order transition, recent-revisit, and history-category signals. B0-B3 and three ablations use
one frozen scoring contract and deterministic POI-ID tie-breaking.

The evaluator predicts an entire target-blind batch before labels are attached. It reports
Hit@1/5/10, MRR, NDCG@5/10, candidate Recall@50/100, macro-category metrics, coverage, validity,
slices, and failure cases. Data, bundle, release, sample-set, and evaluation hashes are logged to
local MLflow with explicit `production_current` and `rerun` lineage.

The online path loads only a verified bundle and exposes FastAPI health, readiness, version,
recommendation, and aggregate metrics endpoints. Streamlit calls the API over HTTP and contains no
duplicate model logic. Monitoring stores an explicit privacy-safe allowlist and supports normal and
deterministically injected category/history drift replay.

## Verified evidence

- Synthetic CI fixture: seven variants completed, with deterministic manifests, reports, and
  MLflow runs; B3 processed eight examples with Hit@10 1.000 and validity 1.000. This tiny result is
  a contract demonstration, not a benchmark claim.
- Local original-split NYC B3: 17,588 examples; Hit@1 0.0706, Hit@5 0.1561, Hit@10 0.1987,
  MRR 0.1078, NDCG@10 0.1294, candidate Recall@100 0.3651, and validity 1.000.
- The NYC evaluator completed in 7.77 seconds on Apple M4/16 GB with Python 3.10.11. Isolated
  in-process prediction latency was p50 0.371 ms and p95 0.418 ms; it is not HTTP, container, GPU,
  or online-production latency.
- Split audit: 83,228/10,339/10,374 train/validation/test events; zero exact-event overlap; temporal
  split boundaries ordered. Existing session overlap was reported and the benchmark was not reset.
- Monitoring: identical 12-event windows produced no alert; injected category/history drift
  reproducibly alerted without using labels or claiming accuracy.

## Production boundaries

Online request schemas reject target, label, and result fields. Known taxonomy may span splits,
but POI/user encoders and all frequency-based signals are train-only. Raw NYC, user trajectories,
coordinates, model weights, experiment state, MLflow stores, and monitoring events remain local and
ignored; the repository tracks only a privacy-safe synthetic fixture and aggregate evidence.

CPU-only non-root Dockerfiles, Compose services, health checks, volumes, secrets checks, and CI are
statically validated. Docker Desktop was unavailable during implementation, so dynamic Compose
startup is explicitly `DEFERRED`. Full-GPU loading, training, parity, latency, and memory validation
are out of scope until complete weights and an independent Linux/NVIDIA environment are available.
