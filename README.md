# duck_gym

面向 [Pollen Robotics MicroDuck](https://pollen-robotics.com/microduck/) 的自研物理仿真与强化学习训练引擎，接入 rsl-rl。

采用原生 C++/CUDA 刚体 AVBD，支持 CPU 求解、GPU 并行环境及完整资产回放。在 RTX 5880 Ada 的 4096 env 共同模型基准中，自研吞吐约 **42.1 万物理 env-step/s**，为 Newton 刚体 AVBD 较快控制组的 **1.57×**。这是固定 HOME 的纯引擎测量，尚未建立等精度比较；RTX 4090 实测待补充。热身、初始化、逐档结果和数值差异见 [性能报告](docs/PERFORMANCE.md)。

- [项目背景](docs/PROJECT_CONTEXT.md)：目标与长期方向。
- [开发路径](docs/DEVELOPMENT_ROADMAP.md)：开发顺序、进度与验收。
- [技术设计](docs/TECHNICAL_PLAN.md)：AVBD 数学、CPU/CUDA 实现、训练接口与可视化。
- [协作与运行规则](AGENTS.md)：文档维护、本地运行、远端同步与构建。
- [资产清单](docs/microduck_asset_manifest.json)：上游版本、模型与网格元数据。
