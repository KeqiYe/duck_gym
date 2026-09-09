# MicroDuck 训练引擎：项目背景

本文件只记录稳定的项目背景与总体约定。开发顺序、阶段进度、待办和验收要求统一放在 [开发路径](DEVELOPMENT_ROADMAP.md)。

## 项目目标

为 [Pollen Robotics MicroDuck](https://pollen-robotics.com/microduck/) 自研物理仿真与强化学习训练引擎，使用官方开源机器人资产，接入 **rsl-rl** 训练策略，使用 **Git** 管理代码与文档。

## 核心技术方向

- 物理求解主线采用 **刚体 AVBD**，以最大坐标、六维刚体局部块和增广拉格朗日处理关节、接触与摩擦。
- 使用原生 **C++/CUDA**，同时提供完整 **CPU 求解**。
- `num_envs` 表示并行独立仿真环境的个数；环境数量、求解后端和渲染开关分别配置。
- GPU 多 env 训练无窗口运行。可视化只需在推理或单 CPU env（`num_envs=1`）下实现，支持 Mac 本地查看，并完整保留 MicroDuck 的可识别外观。

## 开发与运行环境

本机为 Apple Silicon Mac，纯 CPU 代码、本地推理和可视化可在本地编译运行。GPU/CUDA 代码在 SSH 远端编译运行：

| 优先级 | 主机 | 工作根目录 |
| --- | --- | --- |
| 首选 | `master172` | `/public/yekq6Data/codex` |
| 回退 | `delltower` | `/home/yekeqi/Documents/HDD1/codex` |

每次远端运行前，先用 **scp/rsync 同步本次代码、配置与必要资产，再在目标主机编译运行**。各工作根目录下使用 `duck_gym/source/`、`duck_gym/build/`、`duck_gym/runs/` 分别存放源码副本、构建产物与模拟结果。

## 文档入口

- [开发路径](DEVELOPMENT_ROADMAP.md)：实施顺序、进度、待办和验收。
- [技术路线](TECHNICAL_PLAN.md)：模块职责、训练接口和本地可视化设计。
- [AVBD 求解器](ARTICULATION_SOLVER.md)：数学结构与求解约定。
- [运行约定](REMOTE_EXECUTION.md)：本地/远端执行与文件同步规则。
- [资产清单](microduck_asset_manifest.json)：官方资产来源与版本。
