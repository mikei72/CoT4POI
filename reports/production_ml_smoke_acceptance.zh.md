# CoT4POI Production ML smoke-first 最终验收

> 验收日期：2026-07-18
> 动态验收 profile：deterministic CPU `smoke`
> 冻结范围：`reports/implementation_handoff.zh.md` 与
> `reports/production_ml_refactor_plan.zh.md`
> 结论：**11 PASS / 0 FAIL / 1 DEFERRED**

## 1. 总结

本轮已完成数据契约、原始 split 审计、train-only CPU baseline、B0-B3/ablation、bundle、
离线评估、本地 MLflow、FastAPI、Streamlit、privacy-safe monitoring、Docker/Compose 静态
配置、CI、README 和可追溯报告。完整 NYC 仅从本地忽略路径读取，仓库新增数据只有无
真实身份和无坐标的 synthetic fixture。

动态 Compose 是唯一 `DEFERRED`：当前环境没有 Docker CLI，用户将在最终验收时安装
Docker Desktop。该项没有被记为通过。full-GPU 动态加载、训练、推理、parity、latency 和
显存结果均未执行、未实现、未声称；静态扫描如实保持 `runtime_status=static_only`、
`dynamic_load_verified=false`。

## 2. 冻结验收清单

| # | 验收项 | 状态 | 证据 |
|---:|---|---|---|
| 1 | CPU smoke bundle 可训练/加载；GPU manifest 如实 static-only | PASS | bundle round-trip/tamper tests；28 个 present file entries、6 个 missing requirements |
| 2 | 论文归档与当前新评估来源分离 | PASS | MLflow `production_current/rerun`；未把论文数字导入为 rerun |
| 3 | production candidate 不使用 target category/POI | PASS | online/offline Pydantic 类型分离；target/label/result 422；label mutation regression |
| 4 | 原始 split 的 event/session/time/sample-ID 审计可复现 | PASS | data manifest、split audit、固定 sample-ID preimage/hash tests |
| 5 | B0/B1/B2/B3 与 CPU ablation 有 MLflow run | PASS | 最终 synthetic experiment 7/7 runs；每个 run 有独立 config fingerprint 和共同 sample-set hash |
| 6 | Top-K、MRR、NDCG、macro、candidate recall、latency 可复现 | PASS | deterministic core hashes；aggregate reports 与本地 artifact 逐值对照通过 |
| 7 | `docker compose up --build` 启动 API/demo/MLflow | **DEFERRED** | Docker Desktop/CLI 尚未安装；静态 Compose 检查 9/9 通过 |
| 8 | README 明确 full-GPU 未实现/验证 | PASS | README 和英文摘要明确 static-only、无 GPU 性能声称 |
| 9 | code/data/model/API/leakage/Docker 测试有汇总 | PASS | pytest 141/141；静态部署 9/9；其余门禁见第 3 节 |
| 10 | monitoring normal/injected replay | PASS | normal 无告警；注入 category/history drift 可复现告警 |
| 11 | README 不虚构线上流量、A/B、持续训练或高可用 | PASS | provenance/truthfulness grep 和独立 Trellis review 通过 |
| 12 | 对外数字只使用实际验收结果 | PASS | README/英文摘要数字与 final aggregate artifacts 逐值核对；未修改或生成简历 |

## 3. 自动化与动态验证结果

| 验证 | 通过 | 失败 | 备注 |
|---|---:|---:|---|
| Pytest 全仓 | 141 | 0 | 1 条第三方 Starlette/TestClient 弃用警告，不影响结果 |
| Ruff | 1 | 0 | `src tests scripts` 全通过 |
| Compileall | 1 | 0 | `src tests scripts` 全通过 |
| `pip check` | 1 | 0 | project-local `.venv` 无 broken requirements |
| `git diff --check` | 1 | 0 | 无 whitespace error |
| Report JSON parse | 1 | 0 | `reports/**/*.json` 全部可解析 |
| Docker/Compose 静态子检查 | 9 | 0 | context、paths、volumes、secrets、ports、health、CI parity |
| Trellis context files | 2 | 0 | implement/check JSONL 各 10 条有效 entry |
| Synthetic variant runs | 7 | 0 | B0-B3 + 3 ablations，均写入 bundle/report/release/MLflow |
| Local NYC B3 run | 1 | 0 | 17,588 examples，原始 split，不提交 raw/trajectory |
| FastAPI 五端点 smoke | 1 | 0 | verified B3 bundle，recommendation_count=5 |
| Monitoring replay | 2 | 0 | normal no-alert；injected alert |
| Docker Compose 动态启动 | 0 | 0 | **1 DEFERRED** |

最终质量命令：

```bash
.venv/bin/ruff check src tests scripts
.venv/bin/python -m compileall -q src tests scripts
.venv/bin/python -m pytest -q
.venv/bin/python scripts/static_deploy_check.py
.venv/bin/python -m pip check
git diff --check
```

## 4. 实际运行证据

### Synthetic final matrix

- dataset/data manifest：
  `c59fe45b070a504880df6791388f0232fd1b260313c90244fa5caf76534e332e`
- 7/7 variants 完成；全部使用 `production_current/rerun` lineage。
- B3 core：
  `fc2e596dd67a05950ac8238ef6f20fe9b7e4233afe9edeb3cd1210ce7d6e003f`
- 详细 aggregate：`reports/evaluation/synthetic_smoke_matrix.json`。
- fixture 只有 8 个评估样本，其指标用于闭环/契约验证，不作为论文 benchmark 结论。

### Local NYC B3

- 输入行数：train 83,228；validation 10,339；test 10,374。
- 精确 event overlap：train-validation/train-test/validation-test 均为 0。
- session overlap：7/0/10；时间边界有序；按冻结要求未重新划分数据。
- 评估样本：17,588；训练 catalog：4,980 POI。
- core SHA-256：
  `a3775a00e73743d8aa56b68b092bb4c864e18ee408090ec4d5b3a39930d78471`。
- Hit@1/5/10：0.0706/0.1561/0.1987；MRR 0.1078；NDCG@10 0.1294；
  Candidate Recall@100 0.3651；validity 1.000。
- Apple M4/16 GB、Python 3.10.11 上 evaluator 用时 7.77 s；in-process predictor
  p50/p95 0.371/0.418 ms。它们不是 HTTP、容器、GPU 或真实线上 latency。
- 详细 privacy-safe aggregate：`reports/evaluation/nyc_local_smoke_summary.json`。

### Full-GPU static inventory

- backend：`full-gpu`；runtime：`static_only`；dynamic verified：`false`。
- 33 个 manifest file entries 中 28 present、5 missing；另有 1 个缺失 DeepSpeed wildcard
  requirement，共 6 个 missing requirements。
- 未 import/load 模型，未运行训练/推理。
- 摘要：`reports/evaluation/full_gpu_static_manifest_summary.json`。

## 5. 泄漏、隐私与兼容裁决

- 在线 `RecommendationRequest` 使用 `extra=forbid`，历史 1-128、Top-K 1-100，禁止所有
  target/label/result 扩展字段。
- `CandidateIndex.fit()` 只接受 train events；完整 taxonomy 可以跨 split，POI/user mapping
  与 popularity/time/transition/category counts 只能由 train 产生。
- evaluator 先完成全批 target-blind snapshot，再附加 label。Recall@50/100 只需要 Top-100；
  trace 保留 Top-100 但用 `candidate_count` 报告完整 catalog，避免 NYC 约 8,760 万候选对象
  的持久内存风险。
- 超过 128 的 legacy session 使用最近 128 条历史，sample ID、split 和 target 不变。
- monitoring 只允许 opaque request/version、bucket/count/histogram/entropy/latency/status；
  category 使用稳定 hash bucket，不保存 user、完整历史、坐标、target 或 accuracy。
- `datasets/`、根 `/model/`、`experiment/`、`artifacts/`、`mlruns/` 和 monitoring events 均被
  Git 忽略；`tests/model/` 已验证不会被 `/model/` 规则误忽略。

## 6. DEFERRED 与未完成项

### 唯一验收 DEFERRED：Docker Compose 动态闭环

安装 Docker Desktop 后执行：

```bash
cp .env.example .env
# 在 .env 中设置：
# NEXT_POI_BUNDLE_HOST_PATH=./artifacts/smoke-final/b3/bundle
docker compose config
docker compose up --build -d
curl -fsS http://127.0.0.1:8000/ready
curl -fsS http://127.0.0.1:8501/_stcore/health
curl -fsS http://127.0.0.1:5000/health
```

随后按 README 的 request 调用 `/recommend`，并确认 demo 能通过 HTTP 获取同一 API 结果。
上述命令尚未在本环境运行，因此当前不能改记为 PASS。

### 当前范围外

- full-GPU 需要补齐 base/Fine 主权重、可选 trainable params/DeepSpeed state，并在独立
  Linux/NVIDIA 环境重新实施动态 backend、parity、latency 和显存验收。
- 未执行 `git add` 或 commit；`tests/model/` 与其余新增文件均为可追踪但待用户审阅的工作树
  变更。这不影响本地实现验收，但提交时必须一并纳入。

## 7. 已解决的实施问题

- project-local editable install 初次受 build isolation/缺 wheel 阻塞；已把 setuptools/wheel
  锁定并使用 `--no-build-isolation`，最终安装和 `pip check` 通过。
- NYC legacy session 曾超过在线 history 上限；已改为稳定保留最近 128 条并加入回归测试。
- NYC 初版 evaluator 会长期保留全 catalog candidate objects；已改为向量化全量计分和
  bounded Top-100 snapshot，指标契约和 label 隔离不变，最终 NYC run 成功。
- 已删除本轮产生且被最终 run 取代的 `artifacts/nyc-smoke` 不完整本地输出；最终
  `artifacts/smoke-final`、`artifacts/nyc-smoke-final`、`artifacts/monitoring-final` 保留在
  ignored 本地目录中。
