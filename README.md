# duck_gym

面向 [Pollen Robotics MicroDuck](https://pollen-robotics.com/microduck/) 的自研机器人仿真与强化学习训练引擎，接入 **rsl-rl**。

项目当前处于工作上下文建立阶段，尚未实现训练功能。

- [工作上下文](docs/PROJECT_CONTEXT.md)：目标、现状、待定问题和决策记录。
- [技术路线](docs/TECHNICAL_PLAN.md)：原生 CPU/CUDA 后端、训练接口与完整可视化。
- [远程运行](docs/REMOTE_EXECUTION.md)：优先 `gpu`，不可连接时回退 `delltower`。
- [资产清单](docs/microduck_asset_manifest.json)：已核查的上游版本与网格引用；网格尚未导入。
- [协作入口](AGENTS.md)：后续工作开始时的阅读顺序和上下文维护约定。

已确定采用原生 **C++/CUDA**，同时提供完整 **CPU 求解**；使用 Git 管理，完整显示 MicroDuck 外观。多厂商 GPU 暂缓。

[Articulation 数学建议](docs/ARTICULATION_SOLVER.md)：约化坐标、Featherstone 空间代数、CRBA/RNEA、小型矩阵分解和柔性接触；尚未实施或定案。

本机为 Apple Silicon Mac，负责编辑与可视化；仿真和训练运行在 SSH 远端。目前 `delltower` 可达并配备 RTX 4090，`gpu` 不可达。首个训练行为尚待确定。
