# 运行正式实验并生成简单比较报告

Status: ready-for-agent

在 CUDA 上通过全部检查与 smoke 后运行 seed 0 两小时训练，提交可复现结果并与既有 PPO 指标并列展示。

## Acceptance Criteria

- 最终评估请求 1,000 局且双方身份各半。
- config、metrics、summary、曲线和比较报告相互一致。
- 报告明确不作算法级结论。
