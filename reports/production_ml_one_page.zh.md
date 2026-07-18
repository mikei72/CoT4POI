# CoT4POI 生产化重构：确定性 Next-POI CPU Smoke 系统

## 项目目标

本项目将以论文实验为中心的 Next-POI 推荐代码，重构为一套可复现、可测试、可服务、
可观测的单机 Production ML 系统。当前唯一通过动态验收的是确定性 CPU smoke 后端。
由于本机缺少完整的 Fine/QLoRA 主权重，旧 GPU 资产只做静态清点和契约校验；项目没有
实现或声称任何 full-GPU 推理、训练、时延、显存或效果结果。

## 系统实现

离线链路显式读取 train、validation、test 文件，保留原始划分，不重新切分数据。系统
完成事件标准化、split 重叠与时间边界审计，并用 SHA-256 为样本生成稳定身份。完整类别
体系可以跨 split 复用，但 POI、用户编码以及流行度、时段、类别和转移统计全部只在
train 上拟合。

CPU baseline 包括全局流行度 B0、加入时间信号的 B1、加入一阶 POI 转移的 B2，以及
结合近期回访和历史类别信号的 B3，同时提供三个消融版本。所有变体共用冻结的计分公式，
分数相同时按 POI ID 稳定排序。评估器先完成整批 target-blind 预测，再接触真实标签，
输出 Hit@1/5/10、MRR、NDCG@5/10、Candidate Recall@50/100、宏观类别指标、覆盖率、
有效性、切片和失败样本分析。数据、bundle、release、样本集合和评估结果均有独立 hash，
并以 `production_current`、`rerun` 标记写入本地 MLflow。

在线链路只加载通过 manifest、配置、文件 hash 和索引不变量校验的 bundle。FastAPI 提供
健康检查、就绪检查、版本、推荐和聚合指标五个端点；Streamlit 只通过 HTTP 调用 API，
不复制推荐逻辑。监控事件采用严格白名单，只保留版本、分桶、计数、类别 hash、候选来源、
熵、时延和状态，并支持正常窗口与人工注入类别/历史长度漂移的离线回放。

## 实测结果

- Synthetic CI fixture：B0-B3 和三个消融共 7 个变体全部完成，均生成确定性 manifest、
  report 和 MLflow run。B3 在 8 个合成评估样本上的 Hit@10 和 validity 均为 1.000。
  该数据仅用于验证系统闭环和契约，不代表正式 benchmark 水平。
- 本地原始 split NYC B3：共评估 17,588 个样本，Hit@1 为 0.0706，Hit@5 为 0.1561，
  Hit@10 为 0.1987，MRR 为 0.1078，NDCG@10 为 0.1294，Candidate Recall@100
  为 0.3651，validity 为 1.000。
- 在 Apple M4、16 GB 内存、Python 3.10.11 环境中，NYC evaluator 用时 7.77 秒；
  单进程内部预测时延 p50 为 0.371 ms，p95 为 0.418 ms。这些数字不等同于 HTTP、
  Docker、GPU 或真实线上服务时延。
- Split 审计读取到 83,228/10,339/10,374 条 train/validation/test 事件，精确事件重叠
  为 0，时间边界有序。原数据中已有的 session 重叠被如实记录，没有通过重切数据掩盖。
- 监控回放中，相同的 12 条参考/当前事件没有触发告警；注入类别与历史长度漂移后稳定
  触发告警。整个过程不使用在线标签，也不把漂移指标表述成准确率。

## 生产边界

在线请求 schema 拒绝 target、label、result 等字段，候选生成与排序接口不能接收真实
下一 POI 或类别。完整 NYC、用户轨迹、坐标、模型权重、实验状态、MLflow store、监控
事件和运行 artifacts 均留在本地忽略目录；仓库只提交无隐私 synthetic fixture 和聚合证据。

CPU-only Dockerfile、API/demo/MLflow Compose 服务、非 root 用户、healthcheck、volume、
secret 检查和 CI 已完成静态验证。实施环境尚未安装 Docker Desktop，因此动态
`docker compose up --build` 明确记为 `DEFERRED`，而不是通过。待补齐主权重并具备独立
Linux/NVIDIA 环境后，full-GPU backend、parity、时延和显存验证才能作为新的实施任务开展。
