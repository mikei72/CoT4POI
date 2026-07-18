# CoT4POI / Next-POI Production ML 重构方案

> 状态：方案设计完成，尚未开始业务代码重构
> 日期：2026-07-18
> 目标：把离线论文原型改造成可复现、可测试、可部署、可观测的单机 Production ML 项目；当前以 CPU smoke 闭环为可执行验收，GPU 权重仅保留静态契约。

## 1. 结论与推荐路线

推荐采用“**论文版外部归档、当前仓库直接生产化、smoke-first**”的路线：

1. 上级 `../CoT4POI` 继续作为完整论文流程归档；当前 `CoT4POI-public` 不再复制一套 legacy 实现。
2. 当前仓库保留原始 benchmark split，增加 split 审计、manifest、稳定 sample ID，并修复候选生成读取真实标签的问题。
3. `smoke` 使用真实 CPU deterministic baseline（time-aware popularity + transition + recent-history/category signals），而不是硬编码或随机 mock。
4. 完整 NYC 数据通过本地 volume 使用；公开仓库另带无真实用户/位置的 synthetic tiny fixture，供 CI 和一键演示。
5. 7B/Fine 权重不搬到当前电脑；只保存已确认的模型结构、artifact manifest 和未来 backend 接口，不实现或声称 full-gpu 已验证。

这条路线能够同时满足：

- 由上级论文仓库保留论文结果和旧权重证据；
- 不把带泄漏的旧结果包装成 production 指标；
- 先交付可运行的 MLE 系统闭环；
- 所有新增 pipeline、指标、API、延迟和监控证据都能在当前电脑实测。

## 2. 当前项目审计结果

### 2.1 已有能力

- 主链已覆盖 raw preprocessing、时空语义、preference、macro、fine reranking、final SFT、QLoRA 和最终评估。
- 本地归档 `legacy_research/ablation_study/` 已有配置、artifact resolution、variant registry、ranking metrics、dry-run 和结果汇总基础，可作为历史参考。
- 本地归档 `legacy_research/environment.yml` 声明了旧 GPU/研究环境，但不属于 Production ML 依赖契约。
- 当前 public NYC 已有约 186 MB 完整代表链路：raw、preprocessed、preference、macro、fine、label encoding 和 final SFT 均存在。
- 同级 `../CoT4POI/datasets` 已保存 CA/TKY canonical raw；不需要从另一台电脑重复搬这两份 raw，历史中间产物仍需补齐。
- base 目录已保存 config/tokenizer/index，缺索引声明的两个约 13.48 GB 权重分片。
- 当前 `experiment/checkpoint-n` 已确认是 CA `checkpoint-13500`：约 16 MB LoRA adapter 主体完整；缺失的 DeepSpeed `global_step13500/` 不阻塞 adapter-only 推理，但阻塞完整恢复训练和 embed/norm 参数恢复。
- CA Fine `best_model` 已有 config/tokenizer，缺少 `model.safetensors` 或 `pytorch_model.bin`。

### 2.2 当前缺口

| 维度 | 当前状态 | Production ML 最小补齐项 |
|---|---|---|
| 数据 | 固定相对路径、无 manifest、按位置 merge | schema、稳定 sample ID、hash、原始 split 审计、train-only 统计 |
| 实验 | 日志和消融输出分散 | MLflow run、artifact/version、统一 evaluator |
| 模型 | macro/fine/final loader 契约不同，主权重不在本机 | 真实 CPU baseline bundle + GPU artifact 静态 manifest validation |
| Serving | 仅 batch 脚本，final 只返回采样 Top-1 | Predictor 抽象、target-blind CPU candidate/ranker、Top-K API |
| 测试 | 无 `tests/`、无 CI | code/data/model/API/tiny pipeline 测试 |
| 部署 | 无 Docker/Compose；本机尚未安装 Docker | CPU smoke Compose；full-gpu 仅文档化接口 |
| 监控 | 无在线事件或 drift 报告 | privacy-safe JSONL/SQLite、延迟/错误/分布回放 |
| 文档 | research prototype README | 架构、trade-off、成本/延迟、泄漏防护、复现实证 |

### 2.3 必须正面处理的风险

1. **Fine target leakage**：本地归档 `legacy_research/cot/4_run.py` 从 `result.category_name` 读取真实目标类别并强制加入 fine candidates；该信号随后进入 final SFT。
2. **Split 可解释性**：TKY/CA 先按 check-in 切分，再生成 session ID；按用户决定保留原始 split，但必须报告跨 split session/event 审计结果。
3. **全量统计**：固定完整 category taxonomy 已由任务设定为预先已知，可以跨 split；频次、流行度、转移概率等可学习统计仍应只由 train 产生。
4. **位置对齐**：semantic/final merge 与 resume 只比较数组长度或下标，没有 sample identity。
5. **权重契约**：final adapter 强依赖 exact base、tokenizer、special tokens、词表和量化配置；推理还硬编码 `32001`。
6. **额外参数风险**：训练允许 `embed,norm` 参与更新，但训练端没有明确生成推理端期待的 `trainable_params.bin`。
7. **环境不可复现**：代码使用新版 `AsyncOpenAI`，环境却锁定 `openai==0.28.1`；CA/图依赖也未完整声明。
8. **文档路径漂移**：旧 README 的 final SFT 路径与本地归档 `legacy_research/merge_for_sft.py` 实际输出目录不一致。

## 3. Trellis 评估

### 决策：实施阶段已迁移到本地 Trellis

原因：

- 初始方案先通过独立规划文件冻结，进入实施阶段后已完整迁移到 Trellis PRD、design、implement 和 research 记录。
- 对外事实源保留在 `reports/implementation_handoff.zh.md`、本文和最终验收报告中，不依赖个人会话日志。
- `.trellis/`、`.agents/`、`.codex/` 作为本机个人工具保持 Git ignored，不进入公开仓库。
- 旧的三份 `planning-with-files` 日志已被正式记录覆盖并删除，避免过期阶段状态误导后续实施。

若未来启用，应迁移现有规划记录，不要同时维护 Trellis 与另一套 `task_plan/findings/progress`。

## 4. 目标系统架构

![Production ML architecture](figures/production_ml_architecture.drawio.png)

可编辑源文件：`reports/figures/production_ml_architecture.drawio`。

### 4.1 在线路径

```text
RecommendRequest
  -> request/schema adapter
  -> preference cache / deterministic fallback
  -> target-blind CPU candidate generator
  -> popularity + transition + time/history scorer
  -> deduplicate/validate Top-K
  -> RecommendResponse
```

### 4.2 离线路径

```text
legacy/raw assets
  -> manifest + schema validation
  -> preserve original split + overlap/time audit
  -> production schema adapter
  -> fixed taxonomy + train-only statistical features/candidate index
  -> baseline/train/evaluate CLIs
  -> MLflow run + versioned release bundle
```

### 4.3 当前运行 profile 与未来接口

| Profile | 用途 | 运行要求 | 真实性边界 |
|---|---|---|---|
| `smoke`（唯一验收 profile） | 一键演示、CI、真实 CPU baseline、接口和监控验证 | synthetic tiny fixture 或本地 NYC volume | 指标与延迟均可本机复现；不声称是 QLoRA 性能 |
| `full-gpu`（未来接口） | 描述 macro/fine/final artifact contract | 当前不运行、不打包主权重 | 仅静态结构与接入点，不声称可加载、可部署或有 GPU 指标 |

## 5. 建议仓库结构

保留现有脚本位置，先通过 wrapper 渐进迁移，不立即移动或重命名研究代码。

```text
CoT4POI-public/
├── configs/
│   ├── data/                 # split、session、schema、candidate 配置
│   ├── models/               # macro/fine/final artifact refs
│   └── profiles/             # smoke runtime + future GPU manifest schema
├── src/nextpoi/
│   ├── contracts/            # request、labeled example、manifest、release schema
│   ├── data/                 # ingest、validate、sessionize、split、sample ID
│   ├── features/             # temporal、preference、macro、fine adapters
│   ├── candidates/           # train-only index、召回、合并、去重
│   ├── models/               # baseline、legacy loader、QLoRA scorer
│   ├── training/             # CLI wrapper、MLflow tracking
│   ├── evaluation/           # ranking、slices、latency、leakage audit
│   ├── serving/              # FastAPI、schemas、predictor lifecycle
│   └── monitoring/           # events、aggregates、drift、failure cases
├── tests/
│   ├── unit/
│   ├── data/
│   ├── model/
│   ├── integration/
│   └── fixtures/
├── reports/
│   ├── figures/
│   ├── evaluation/
│   └── monitoring/
├── artifacts/                # ignored；manifest/release bundle 的本地挂载点
├── legacy_research/          # ignored；本地旧论文代码归档，不进入公开仓库或生产镜像
├── pyproject.toml             # CPU service/dev 环境
├── Dockerfile.api
└── docker-compose.yml
```

顶层继续使用 `datasets/`，不改成 `data/`。原因是所有旧脚本和已有中间产物都绑定 `datasets/<dataset>/...`；新代码通过配置读取外部 volume，避免复制 147 MB 数据或破坏旧路径。

## 6. 数据流水线设计

### 6.1 正确顺序

用户提出的功能全部保留，并维持原始 benchmark split：

```text
raw ingest
-> schema/time normalization
-> preserve original train/val/test split
-> reproduce original sessionization and ID encoding
-> audit event/session overlap and temporal order
-> load fixed known category taxonomy
-> fit popularity/transition/time-bucket statistics on train only
-> transform train/val/test
-> temporal/preference/macro/fine features
-> target-blind POI candidate generation
-> final SFT/evaluation views
```

关键规则：

- 保留原始 warm-start split，不在本次重构中另建 cold-start 或 session-safe split。
- user/POI 可以跨 split；event/session 交集和时间边界作为审计结果如实报告，不静默重切数据。
- 完整 category taxonomy 是预先已知的闭世界标签空间；encoder/padding 继续复用原 `label_encoding.pkl`。
- popularity/transition/time-bucket 等从数据学习的统计只在 train fit。
- `LabeledExample` 与 `RecommendationRequest` 是不同类型；在线类型不得包含 `result`、`target_category` 或 `target_poi`。
- merge 必须按 `sample_id` join，并核验 upstream hash，禁止只按数组位置 `zip`。

### 6.2 稳定 sample ID

建议由以下字段规范化后计算 SHA-256：

```text
dataset + split_protocol + raw_user_id + session_id
+ target_timestamp_utc + target_raw_poi_id
```

该 ID 贯穿 QA、semantic、preference、macro、fine、final SFT、prediction 和 failure-case report。

### 6.3 Candidate generation

训练集构建 `candidate_index`，至少包含：

- encoded/raw POI ID 对照；
- fine/macro category；
- POI 经纬度或 centroid（若可用）；
- 全局、时间段、weekday/weekend popularity；
- 一阶 POI transition count；
- data/model/schema version。

在线召回取以下并集并记录来源：

1. 用户历史中的高频/近期回访 POI；
2. fine top categories 内的 train-popular POI；
3. 与最后位置邻近的 POI（坐标存在时）；
4. 历史最后 POI 的 transition neighbors；
5. 全局/时间段 popular fallback。

候选阶段独立评估 `Candidate Recall@50/100`，否则最终 ranker 指标无法定位瓶颈。

### 6.4 中间数据复用矩阵

| Artifact | 复用结论 | 推荐动作 |
|---|---|---|
| raw files | 可复用 | 记录来源、许可、行数、SHA-256；保留原 split |
| CA filtered raw | 条件复用 | 补 polygon/subset/env hash；明确实际覆盖 CA+NV |
| external NYC raw split | 可复用 | 检查 event/session 交集和时间边界，但不重切 |
| `label_encoding.pkl` + preprocessed CSV | 必须捆绑复用 | 记录 classes/padding/offset/hash |
| QA / temporal semantic | 条件复用 | 上游 hash 相同即可复用；补 `sample_id` 验证一一对应 |
| preference | 优先复用 | 生成昂贵；记录 provider/model/prompt/temperature，在线采用 cache/fallback |
| macro outputs | 条件复用 | 输入一致时复用；后续 pin 7 类、阈值、Hub revision |
| category map | 可复用 | 作为预先已知固定 taxonomy，保留人工校正 provenance；Count 不进入模型特征 |
| `*_with_fine.json` | **必须重生成** | 改为全类别分批评分，去掉真实 target category 注入 |
| `final_sft_*.json` | **必须重生成** | 继承旧 Fine 输入；改为 sample ID join |
| Fine checkpoint | 优先复用 | 不先重训，使用全类别评分重新评估 |
| final QLoRA checkpoint | 先 compatibility gate | 不先重训，评估修正后 Fine 信号下的质量 |

## 7. GPU 模型静态兼容契约（当前不运行）

### 7.1 Model bundle 分层

不能只记录一个 `model_path`：

```yaml
macro:
  model_id: cross-encoder/nli-roberta-base
  revision: <pinned commit>
  labels_sha256: <hash>
fine:
  model_dir: <full HF sequence-classification directory>
  category_map_sha256: <hash>
  prompt_version: fine_v1
final:
  base_model_ref: <exact local snapshot or Hub revision>
  tokenizer_ref: <exact snapshot>
  adapter_dir: <PEFT directory>
  trainable_params_file: <optional but validated>
  prompt_version: final_v1
  vocab_size: <measured>
  quantization: nf4-double-bf16
mapping:
  label_encoding_sha256: <hash>
  poi_id_offset: <0 for NYC, 1 for TKY/CA>
```

### 7.2 当前静态验证门禁

当前只做不依赖主权重的验证：

1. 记录 base config、tokenizer、权重索引、Fine config/tokenizer 和 adapter 文件清单、大小、SHA-256。
2. 校验模型架构、vocab/special IDs、context、LoRA rank/targets、dataset/checkpoint step 和映射引用。
3. 把缺失的 base/Fine 主权重、`trainable_params.bin`、DeepSpeed states 标记为 `not_present`，不把它们当作 smoke 失败。
4. release manifest 明确 `runtime_status=static_only`、`dynamic_load_verified=false`。
5. README 只引用论文归档指标，不写 GPU latency、显存、parity 或“本机成功加载”。

### 7.3 未来 full-gpu 接入点

未来若在独立 Linux/NVIDIA 环境补齐主权重，可实现第二个 `Predictor` backend，并执行动态加载、target-blind Fine、QLoRA Top-K 和 parity gate。本阶段不编写未经验证的 GPU runtime，也不把它列入当前验收清单。

## 8. Experiment tracking

推荐主追踪器：**本地 MLflow**。

原因：Docker Compose 可一键演示、无需外部账号、适合保存 params/metrics/artifacts/model bundle；W&B 保留为可选 adapter，不在第一阶段维护双栈。

每个 run 至少记录：

- `git_sha`、完整命令、resolved config；
- dataset/split/sample manifest hash；
- smoke ranker/config version、可选 archived GPU manifest hash；
- category map、label encoder、feature/candidate version；
- candidate source 配置与 recall；
- offline metrics、slice metrics、invalid rate；
- cold/warm latency、CPU/RAM/VRAM、artifact size；
- preference API tokens/cost；
- `result_lineage=paper_archive|production_current`；
- `result_source=rerun|thesis_import`。

论文中的既有结果可导入 MLflow，但必须标记 `thesis_import` 与 `reproduced=false`，不能伪装为新环境 rerun。

### 首批实验矩阵

| Run | 系统 | 训练成本 | 目的 |
|---|---|---:|---|
| B0 | global popularity | 无 | 最低基线 |
| B1 | time-aware popularity | 无 | 验证时间特征价值 |
| B2 | transition + category candidate rank | 无 | production smoke baseline |
| B3 | B2 + recent-history revisit/category | 无 | 完整 production smoke ranker |
| A1–A3 | no transition/time/history | 无 | CPU baseline ablation |
| T0 | thesis metrics import | 无 | 标记 `thesis_import`，只作论文证据，不伪装 rerun |

## 9. Offline evaluation

### 9.1 推荐质量

- POI：Hit/Top@1、5、10，MRR，NDCG@5/10；
- Candidate：Recall@50/100；
- Macro：Top-1 accuracy、Hit@3、MRR；
- Fine：Hit@1/5/10、MRR；
- Coverage：unique POI/category coverage、long-tail coverage；
- Validity：invalid ID、duplicate Top-K、empty candidate rate；
- Slice：history length、rare POI、time gap、weekday/weekend、dataset/city。

### 9.2 系统性能

- cold-start model load time；
- warm request p50/p95/p99；
- stage latency：preference/macro/fine/candidate/final；
- fixed concurrency 下吞吐；
- peak RSS、VRAM、model/artifact size；
- preference API token/cost per request。

所有性能数字必须绑定硬件、profile、candidate size、batch size 和模型版本。

## 10. Serving contract

### 10.1 `/recommend`

请求草案：

```json
{
  "dataset": "nyc",
  "user_id": 123,
  "history": [
    {
      "poi_id": "raw-or-encoded-id",
      "category_name": "Coffee Shop",
      "timestamp": "2026-07-17T08:30:00Z",
      "latitude": 40.72,
      "longitude": -74.00
    }
  ],
  "target_time": "2026-07-17T12:00:00Z",
  "top_k": 10,
  "profile": "smoke"
}
```

响应草案：

```json
{
  "recommendations": [
    {
      "rank": 1,
      "poi_id": "decoded-raw-id",
      "model_poi_id": 42,
      "category": "Restaurant",
      "score": -1.23,
      "candidate_sources": ["fine_popular", "transition"]
    }
  ],
  "macro": [{"category": "food & dining", "score": 0.82}],
  "versions": {
    "release": "nyc-production-v2-r1",
    "data": "nyc-v2",
    "model": "legacy-adapter-compatible-r1"
  },
  "latency_ms": {"total": 180.0, "candidate": 4.1, "model": 150.2},
  "request_id": "opaque-id"
}
```

补充最小端点：

- `GET /health`：进程存活；
- `GET /ready`：artifact 已加载且 manifest 通过；
- `GET /version`：release/data/model 版本；
- `GET /metrics`：聚合服务指标，不暴露原始轨迹。

## 11. Docker Compose 与部署

默认 compose：

```text
api       FastAPI，MODEL_BACKEND=smoke
demo      Streamlit/Gradio 调用 API
mlflow    本地 tracking UI + volume
```

验收命令目标：

```bash
docker compose up --build
curl localhost:8000/ready
curl -X POST localhost:8000/recommend ...
```

边界：一键启动证明 CPU baseline、API、artifact contract、Top-K、监控和 demo 的闭环；GPU backend 只有静态 contract，不属于 Compose 服务。

## 12. Monitoring mock

记录 privacy-safe 事件：

- request ID、release/model/data version；
- history length、time gap bucket、unknown ID count；
- candidate count/source distribution；
- macro histogram、Top-K category histogram、score entropy；
- stage/total latency、cache hit、status/error code；
- 不记录明文 user ID、完整轨迹或精确位置。

离线回放报告：

- latency p50/p95/p99、error rate；
- macro/Top-K 分布与 train reference window 的 JS divergence/PSI；
- invalid/duplicate/empty cases；
- 正常 replay 不告警；人工注入 category/history-length drift 后触发告警；
- 只有真实 target 延迟到达时才 backfill accuracy/NDCG，不能把无标签 drift 当作在线质量指标。

第一版使用 JSONL 或 SQLite + Pandas/SciPy 生成静态报告即可；Prometheus/Evidently 是可选增强，不引入 Grafana/Kafka/feature store。

## 13. 测试与 CI

| 测试层 | 核心检查 |
|---|---|
| Unit | time bucket、sample ID、candidate union/dedup、metric、prompt rendering |
| Data | schema、timestamp、event/session split 交集、train-only fit、mapping offset、hash |
| Leakage regression | 在线 request 类型禁止 target；candidate generator 不接受 label；test target 不影响 candidate set |
| Model | smoke bundle load、rank determinism、合法 Top-K；GPU manifest 仅做静态 schema/config 校验 |
| Integration | tiny raw → manifest → candidate → evaluate；API request/4xx/ready/version |
| Docker | 从零 build、health、smoke recommendation |
| Performance | 固定硬件下 cold/warm latency 和资源上限 |

GitHub Actions 只跑 CPU：lint、unit/data tests、tiny pipeline、API smoke。GPU 动态加载和 rerun 不属于当前 CI。

## 14. 四个月实施路线

### 第 1 月：数据、CPU baseline 与评估骨架

**Week 1：Contracts + fixture**

- 定义 request/data/model/release schema；
- 生成无真实用户/位置的 synthetic tiny fixture；
- 为现有附属文件生成 static-only GPU manifest；
- 验收：fixture 可版本化，旧文件零改写，缺主权重不影响 smoke。

**Week 2：Data contracts**

- 稳定 `sample_id`、schema、split/leakage validator；
- 冻结 NYC 当前 schema；
- 修复 README 路径/主链说明的设计稿；
- 验收：fixture 上可重复生成 event/session/time audit；发现交集时如实报告，不改变原 split。

**Week 3：Target-blind data/candidates**

- target-blind POI candidates；
- 固定 category taxonomy + train-only POI/statistical index；
- popularity/transition/time baseline；
- 验收：candidate generation 不读取 label，Recall@K 可计算。

**Week 4：Unified evaluation + MLflow**

- 统一 POI/macro/fine metrics 和 slices；
- 导入论文结果并明确 `thesis_import`；
- 跑 B0/B1/B2；
- 验收：同一 manifest/run 可复现报告。

### 第 2 月：Serving、Docker 与实验追踪

**Week 5：Reusable smoke predictor**

- 统一 `Predictor` 接口；
- 实现 B0/B1/B2/B3 CPU ranker 和 Top-K；
- 生成 versioned smoke model bundle。

**Week 6：FastAPI**

- `/recommend`、health/ready/version/metrics；
- request lifecycle、模型启动加载、错误契约；
- 验收：Top-K 无重复、ID 可解码、非法输入返回 4xx。

**Week 7：Docker Compose + demo**

- smoke 默认 profile、MLflow、demo；
- 验收：全新环境一条命令启动 smoke 闭环。

**Week 8：Latency benchmark**

- stage profiler、cold/warm、candidate size trade-off；
- 固定硬件 latency/cost 报告；
- 验收：README 中每个性能数字可追溯到 run/config。

### 第 3 月：可靠性、监控和 CI

**Week 9–10**

- code/data/model/integration/leakage tests；
- CPU tiny CI；
- release bundle validation 与 rollback smoke。

**Week 11–12**

- privacy-safe event logging；
- normal replay + injected drift；
- failure-case dashboard/report；
- 验收：正常回放不告警，注入漂移可复现触发。

### 第 4 月：优化、证据和 MLE 展示

**Week 13–14**

- 候选规模、batch scoring、cache、prompt length 优化；
- 完整 ablation 与 slice report；
- 完整 CPU ablation、slice 与 failure-case report。

**Week 15–16**

- MLE README、架构图、trade-off、data leakage、latency/cost；
- 一页英文 project summary；
- 录制 demo、整理 OpenAPI/curl、MLflow/monitoring 截图；
- 发布 versioned release bundle 与最终验收报告。

## 15. 独立实施对话与模型分工

当前对话只负责规划。实际编码由用户另开的实施对话作为 orchestrator，且必须先读取 `reports/implementation_handoff.zh.md`，不依赖当前聊天的隐藏上下文。实施 orchestrator 负责：

- 架构、兼容边界、关键代码 review、跨模块验收；
- 合并 subagent 发现并维护唯一规划/决策记录；
- 在数据、模型、serving 三条流发生接口变化时做最终裁决。

建议分工原则：

| 任务 | 适合模型类型 |
|---|---|
| 总体分解、接口冻结、泄漏审计、checkpoint 兼容、跨模块调试、最终 review | GPT-5.6 xhigh |
| 文件 inventory、代码骨架、schema/manifest、指标/测试、API/demo、Docker 配置、报告汇总 | 边界明确的 subagent；不强制指定底层模型，任务必须有验收命令 |

具体实现可交给 subagent，但不把任一 worker 作为唯一的全仓 orchestrator，也不强制路由到特定底层模型。每个 subagent 任务必须写清允许修改的文件、输入输出契约、禁止项和测试命令；完成后由主线程复核 diff、接口和整体验收。

建议并行开发流：

1. Data contracts + manifest；
2. CPU baseline bundle/predictor + GPU static manifest validation；
3. Evaluator/MLflow；
4. Serving/monitoring；
5. 主线程负责架构、集成和最终质量门禁。

多个 agent 不应同时修改共享规划、公共 schema 或依赖锁文件；这些文件由实施 orchestrator 统一合并。完整派工与门禁见 `reports/implementation_handoff.zh.md`。

## 16. 最终验收清单

- [ ] CPU smoke model bundle 可训练/加载；GPU 附属文件 manifest 标记为 static-only；
- [ ] 上级论文归档结果与当前仓库新评估来源标记清楚；
- [ ] 当前生产流程不使用 target category/POI 构造候选；
- [ ] raw event、session、时间和 sample ID 的原始 split 审计可复现并有报告；
- [ ] B0/B1/B2/B3 与 CPU ablation 均有 MLflow run；论文指标只以 `thesis_import` 导入；
- [ ] Top@K、MRR、NDCG、macro accuracy、candidate recall、latency 均可复现；
- [ ] `docker compose up --build` 可启动 smoke API + demo + MLflow；
- [ ] README 明确 full-gpu 未在当前环境实现或验证；
- [ ] 代码、数据、模型、API、leakage、Docker 测试均有通过/失败汇总；
- [ ] monitoring replay 能产生正常报告和漂移告警；
- [ ] README 不虚构真实线上流量、A/B、自动持续训练或高可用；
- [ ] 简历数字只使用实际验收产生的指标。

## 17. 已冻结的环境决策

当前不再要求提供任何主权重、DeepSpeed state 或额外数据。NYC 完整链路足以开发 smoke，CA/TKY raw 足以保留 parser/schema 覆盖。

Docker runtime 已选定为 Docker Desktop。用户将在最终验收时自行安装，因此实施阶段先完成本地 Python 闭环、Dockerfile/Compose 静态检查和文档；动态 `docker compose up --build` 验收明确标记为 deferred，不得伪造为已通过。安装后再执行 Compose 构建、health、API recommendation 和服务联通性验收。

## 18. 参考资料

- [Made With ML GitHub](https://github.com/GokuMohandas/Made-With-ML)
- [Made With ML course](https://madewithml.com/courses/mlops/)
- [Experiment tracking](https://madewithml.com/courses/mlops/experiment-tracking/)
- [Testing ML systems](https://madewithml.com/courses/mlops/testing/)
- [Model serving](https://madewithml.com/courses/mlops/serving/)
- [Monitoring ML systems](https://madewithml.com/courses/mlops/monitoring/)
- [ml-ops.org](https://ml-ops.org/)
- [MLOps principles](https://ml-ops.org/content/mlops-principles)

Made With ML 的课程与核心工程内容主要形成于 2023 年，当前仍在线并适合作为生命周期与工程原则参考；其中 Ray/Anyscale 依赖不应直接当作 2026 年单机项目模板。
