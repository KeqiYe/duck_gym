# duck_gym

面向 [Pollen Robotics MicroDuck](https://pollen-robotics.com/microduck/) 的自研机器人仿真与强化学习训练引擎，接入 **rsl-rl**。

项目当前处于工作上下文建立阶段，尚未实现训练功能。

- [工作上下文](docs/PROJECT_CONTEXT.md)：目标、现状、待定问题和决策记录。
- [技术路线](docs/TECHNICAL_PLAN.md)：多厂商 CPU/GPU 后端、训练接口与完整可视化。
- [远程运行](docs/REMOTE_EXECUTION.md)：优先 `gpu`，不可连接时回退 `delltower`。
- [资产清单](docs/microduck_asset_manifest.json)：已核查的上游版本与网格引用；网格尚未导入。
- [协作入口](AGENTS.md)：后续工作开始时的阅读顺序和上下文维护约定。

已确定 CPU/CUDA 支持、多厂商 GPU 优先、Git 管理，以及能一眼认出 MicroDuck 的完整外观显示要求。技术建议为 C++ + Kokkos，须先通过工具链与 PyTorch 互操作验证，必要时回退原生 C++/CUDA。

本机为 Apple Silicon Mac，负责编辑与可视化；仿真和训练运行在 SSH 远端。目前 `delltower` 可达并配备 RTX 4090，`gpu` 不可达。首个训练行为尚待确定。
