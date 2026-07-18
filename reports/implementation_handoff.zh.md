# CoT4POI Production ML 实施交接

> 状态：规划冻结，可交给全新的 Codex 对话执行。
> 项目路径：`<repo-root>`
> 本文是实施阶段的事实源；聊天记忆只能辅助，不能覆盖本文与仓库实时证据。

## 1. 新对话的职责

新对话担任实施 orchestrator：完成代码改造、测试、文档和验收记录。开始编码前依次执行：

1. 读取当前生效的用户/项目 `AGENTS.md` 与已触发 skill；
2. 完整读取本文；
3. 完整读取 `reports/production_ml_refactor_plan.zh.md`；
4. 检查 `git status --short`，保护所有既有用户改动；
5. 以真实代码、数据和日志复核即将修改的调用链；
6. 建立实施计划，再按本文阶段门禁推进。

不要假设能自动继承规划对话的完整上下文。若实时仓库与本文冲突，以实时证据为准，但必须先记录冲突、影响和处理决定。

## 2. 总目标与当前验收边界

把现有 CoT4POI 研究代码改造成可展示的单机 Production ML 项目，打通：

```text
data contract -> train-only feature fit -> CPU baseline -> offline evaluation
-> MLflow tracking -> FastAPI -> Streamlit demo -> monitoring replay -> Docker Compose
```

当前机器不具备完整 GPU 权重和 full-gpu 运行条件，因此唯一动态验收 profile 是真实、确定性的 CPU smoke。GPU 相关只交付静态 artifact manifest、配置校验和未来 backend 接口，不实现或声称已经验证 QLoRA/Fine 推理、GPU latency、显存占用或 parity。

Docker runtime 已选定为 Docker Desktop。用户将在最终验收时自行安装；在此之前完成 Dockerfile/Compose 和静态检查，并把动态 Compose 验收标记为 `DEFERRED`，不能记为通过。

## 3. 已冻结的产品与数据决定

- 当前 public 仓库只维护 production 流程；上级 `../CoT4POI` 保留完整论文版流程。
- 保留原始 train/val/test split，不重分数据；新增 split audit 和 manifest。
- 已知完整 category taxonomy 可以跨 split 复用；频率、流行度、转移概率等统计只能由 train 拟合。
- 公开仓库提交 synthetic tiny fixture，禁止包含真实用户 ID、精确坐标或可反推轨迹；完整 NYC 数据通过本地路径或 volume 使用，不提交。
- smoke 必须是真模型逻辑，不允许 hard-coded/random mock：默认由 time-aware popularity、first-order POI transition、recent-history revisit/category signals 组成。
- experiment tracking 使用本地 MLflow；不同时维护 W&B。
- 服务使用 FastAPI，demo 使用 Streamlit，监控第一版使用 privacy-safe JSONL/SQLite 与离线聚合报告。
- 不引入 Ray、Anyscale、Kubernetes、Kafka、feature store、Prometheus/Grafana 或自动持续训练。
- `reports/` 纳入版本控制；所有性能数字和图表必须可追溯到实际 run/config/artifact。

## 4. 必须保护的兼容性与泄漏边界

### 4.1 资产现状

- base model 名称：`Llama-2-7b-longlora-32k-ft`；本机只有 config/tokenizer/index，缺约 13.48 GB 的两个主权重分片。
- `experiment/checkpoint-n/` 保留 LoRA adapter 与 trainer 状态，但缺远端 `global_step*` DeepSpeed 状态。
- CA Fine `best_model/` 有 XLM-R 配置/tokenizer，缺主 weights 文件。
- `datasets/nyc` 是完整代表数据链路，可用于 schema、split audit 和本地 baseline；不能据此声称 CA/TKY 的全部派生产物已兼容。

### 4.2 必须建立的回归保护

- 本地归档 `legacy_research/cot/4_run.py` 的 Fine 候选构造强制加入真实 target category，存在标签泄漏；production candidate generator 的签名和数据类型不得接收 target/label。
- 训练样本类型可以含 label，在线 `RecommendationRequest` 必须从 schema 层禁止 target/label/result 字段。
- 所有跨阶段 merge 使用稳定 `sample_id`，不能按列表下标或隐含文件顺序对齐。
- 固定 taxonomy 不等于可复用全数据统计；所有可学习统计必须有 train-only fit 测试。
- Fine 全类别评分若未来恢复，应使用 CPU 类别表、GPU micro-batch、CPU Top-K 汇总，并修复 CSV header 类别、`set` 非确定顺序和分数校准问题。
- NYC `label_encoding.pkl` 存在 scikit-learn 版本漂移，需导出稳定 JSON/CSV sidecar，不以跨版本 pickle 作为长期契约。
- 旧中间产物与当前 `max_result_count` 等配置存在漂移，只能作为兼容样本，不能宣称由当前脚本完全复现。

## 5. 建议目标结构

在尊重仓库现状的前提下采用最小结构，不机械照搬模板：

```text
configs/
src/next_poi/
  data/
  features/
  models/
  training/
  evaluation/
  serving/
  monitoring/
tests/
  fixtures/
reports/
  figures/
  experiments/
  evaluation/
  monitoring/
artifacts/              # 默认忽略运行产物，仅保留 manifest/example
Dockerfile.api
Dockerfile.demo
docker-compose.yml
pyproject.toml          # 若现有依赖方式不适合，再做最小迁移
```

不要为了匹配此树而一次性搬动全部旧脚本。优先在 `src/next_poi/` 中建立稳定接口与 adapter，让旧格式通过显式 reader 接入。

## 6. 实施阶段与硬门禁

### Phase 0：基线与变更边界

工作：盘点现有运行入口、依赖、测试和 dirty files；记录现有 Python 语法/测试基线；冻结第一批接口。

门禁：

- 没有覆盖或格式化无关用户文件；
- 真实失败和环境缺失已记录；
- 完成标准和测试命令写入实施计划。

### Phase 1：契约、fixture 与 manifest

工作：建立 typed/config schema、稳定 `sample_id`、NYC legacy reader、synthetic fixture、split/data/model manifest、encoder sidecar 导出与校验。

门禁：

- fixture 不含真实个人或精确位置数据；
- train/val/test 计数、hash、schema 可复现；
- 在线请求禁止 target 字段；
- 同一输入生成相同 sample ID；
- 缺失 GPU 主权重时静态 manifest 明确返回 `static_only`，不静默降级为已加载。

### Phase 2：CPU baseline、候选与离线评估

工作：实现 train-only fit、确定性 candidate union/ranking、model bundle save/load、batch predictor，以及 Top@K、MRR、NDCG、macro category accuracy、candidate Recall@K 和 slice/failure cases。

建议 baseline：

- B0：global popularity；
- B1：time-aware popularity；
- B2：B1 + first-order transition；
- B3：B2 + recent-history revisit/category signal；
- ablation：逐项移除 time/transition/revisit-category。

门禁：

- candidate generator API 不接收 label；
- 改变 test label 不改变 candidate set；
- 所有统计只由 train 产生；
- save/load 前后 Top-K 一致；
- fixed seed/config 下重复运行指标一致；
- 指标对空列表、重复推荐、短列表有单测。

### Phase 3：MLflow 与可追溯报告

工作：记录 B0-B3 和 ablation 的 params、metrics、data fingerprint、code/config version、latency 与 artifacts；生成对比表和 failure-case 报告。

门禁：

- 每个 run 能追溯 data/model/config；
- 论文旧指标只能以 `thesis_import` 明确标记，不混作新跑结果；
- 没跑出的 latency/cost/accuracy 不写入 README 或简历。

### Phase 4：FastAPI 与 Streamlit

最低端点：

- `GET /health`：进程存活；
- `GET /ready`：bundle 已加载且 manifest 校验通过；
- `GET /version`：release/data/model 版本；
- `POST /recommend`：历史轨迹、target time、Top-K -> POI/category/score/source；
- `GET /metrics`：聚合指标，不暴露原始轨迹。

门禁：

- API 和 batch evaluator 共用同一个 predictor/normalization；
- 正常请求返回不重复、可解码的 Top-K；
- 非法 ID、空历史、非法时间、超限 K 有确定的 4xx/降级契约；
- API 集成测试不依赖 GPU 或完整 NYC。

### Phase 5：监控与回放

工作：记录 request/release/model/data version、history bucket、unknown count、candidate source、category histogram、score entropy、stage/total latency、status/error；生成 normal 与 injected-drift replay。

门禁：

- 不记录明文 user ID、完整轨迹、精确位置；
- 正常 replay 不触发阈值，人工注入 category/history drift 可复现触发；
- 无在线 label 时只报告系统/输入/输出漂移，不把漂移伪装成 accuracy。

### Phase 6：Docker、CI 与 README

工作：CPU-only API/demo/MLflow Compose、healthcheck、环境样例、GitHub Actions、MLE README、架构图和一页英文项目摘要。

门禁：

- 当前先完成 Docker 静态配置、路径和 secret 检查；
- Docker Desktop 未安装时，动态 Compose 检查记为 `DEFERRED`；
- 用户安装后执行 `docker compose up --build`、health、recommend、demo 到 API、MLflow UI 的完整联通性；
- README 明确 local production-like、smoke backend 和 full-gpu 未验证边界。

### Phase 7：最终集成验收

按 `reports/production_ml_refactor_plan.zh.md` 第 16 节逐项核对，汇总通过、失败、deferred 数量。任何失败都写明真实原因，不以跳过代替通过。

## 7. 模型与 agent 派工

### GPT-5.6 xhigh：实施 orchestrator

必须由其主导：

- 阶段分解、公共接口与配置冻结；
- data leakage、split、taxonomy 与旧资产兼容裁决；
- 多模块变更、非局部失败和依赖冲突调试；
- subagent diff review、集成测试与最终验收；
- 是否需要偏离本文的决策。

### 具体实现 worker（模型不强制）

适合：

- 文件/数据 inventory 与 manifest；
- typed schema、reader、fixture 与纯函数；
- baseline/metrics 的局部实现及单元测试；
- FastAPI 端点、Streamlit 页面、监控聚合；
- Docker/Compose/CI 配置；
- 文档表格、OpenAPI/curl 和测试结果整理。

不应单独决定：跨模块架构、泄漏定义、数据重生成、旧权重兼容结论、失败门禁放宽或最终发布。

每个实现任务都必须包含：

```text
目标：一个可独立验收的具体结果
允许修改：精确到目录/文件
输入/输出契约：schema、函数或 endpoint
禁止项：无关重构、真实数据提交、虚构结果等
验收：具体测试/检查命令
交付：diff 摘要、测试通过/失败数量、风险
```

建议并行流：A 数据/契约；B baseline/evaluation/MLflow；C API/demo/monitoring/Docker。公共 schema、依赖锁文件和共享规划由主线程统一修改。并行流开始前先冻结接口，完成后由主线程集成。

## 8. 实施中的停止条件

只有以下情况需要暂停并询问用户：

- 必须更改原始 split、覆盖旧中间数据或重新训练 GPU 模型；
- 必须提交真实 NYC 数据、主权重或敏感轨迹；
- 需要安装/升级全局依赖，或改变用户 Docker/系统配置；
- 实时证据推翻已冻结的核心范围，且不同选项会实质改变交付；
- 需要删除、覆盖或迁移不可恢复资产。

普通实现细节、可逆的局部设计和测试修复由实施 orchestrator 自主决定。

## 9. 最终交付清单

- 可复现 synthetic smoke 与本地 NYC 两种数据入口；
- versioned data/model/release manifest；
- B0-B3、ablation、offline metrics 与 failure cases；
- MLflow runs 与可追溯 report；
- FastAPI、Streamlit、privacy-safe monitoring/replay；
- Dockerfile、Compose、CI 和环境样例；
- 单元/data/leakage/model/integration/API/Docker 验收汇总；
- MLE 风格 README、架构图、一页英文 project summary；
- 对 full-gpu、Docker 动态验收和未完成项的诚实边界说明。

## 10. 新对话启动指令

将下面内容原样作为新对话首条消息；无需复制整个旧对话：

```text
请作为本项目实际实施阶段的 orchestrator，在
<repo-root>
完成 CoT4POI -> Production ML 的 smoke-first 重构。

开始前必须完整读取：
1. 当前生效的 AGENTS.md；
2. reports/implementation_handoff.zh.md；
3. reports/production_ml_refactor_plan.zh.md；
4. git status 与相关真实代码/数据。

以 implementation_handoff 为冻结范围和验收标准，不依赖旧聊天的隐藏上下文。
使用 GPT-5.6 xhigh 负责总体分析、接口、泄漏/兼容裁决、集成和最终 review；
将边界明确、可独立测试的实现任务派给 subagent，不强制指定底层模型。允许按数据/契约、
baseline/evaluation/MLflow、API/demo/monitoring/Docker 拆分 subagent，
但公共 schema、依赖锁和共享规划由主线程统一维护。

当前机器缺完整 GPU 主权重，唯一动态目标是确定性 CPU smoke；不实现或虚构 full-gpu 结果。
Docker Desktop 已选定，但我会在最终验收时自行安装；此前完成本地 Python 闭环与 Docker 静态配置，
将 Compose 动态验证标为 DEFERRED。保留原始 split，完整 NYC 只走本地路径/volume，
仓库仅提交无隐私 synthetic fixture。执行过程中自主推进，每阶段按文档门禁验证，
最后汇总通过、失败、deferred 数量及未完成项。
```

## 11. 参考

- 完整设计：`reports/production_ml_refactor_plan.zh.md`
- 架构源文件：`reports/figures/production_ml_architecture.drawio`
- 架构预览：`reports/figures/production_ml_architecture.drawio.png`
- Codex 模型说明：[Models](https://learn.chatgpt.com/docs/models)
- Codex 子 agent 指南：[Subagents](https://learn.chatgpt.com/docs/agent-configuration/subagents.md)
