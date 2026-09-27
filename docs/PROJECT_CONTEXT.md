# MicroDuck 训练引擎：项目背景

本文件只记录稳定背景。实施顺序见 [开发路径](DEVELOPMENT_ROADMAP.md)，具体设计见 [技术设计](TECHNICAL_PLAN.md)。

## 项目目标

为 [Pollen Robotics MicroDuck](https://pollen-robotics.com/microduck/) 自研物理仿真与强化学习训练引擎，使用官方开源机器人资产，接入 **rsl-rl** 训练策略，使用 **Git** 管理代码与文档。

## 核心方向

- 采用 **刚体 AVBD**，以最大坐标、六维刚体局部块和增广拉格朗日处理关节、接触与摩擦。
- 使用原生 **C++/CUDA**，提供完整 **CPU 求解**；多厂商 GPU 暂不作为当前实施目标。
- 性能优化面向 **MicroDuck 专用引擎**：允许放弃其他机器人的兼容性，以提高吞吐；目标是实测吞吐超过 Newton，比较范围和验证进度见开发路径。
- `num_envs` 表示并行独立仿真环境个数；环境数量、求解后端和渲染开关分别配置。单环境不限制 CPU 核数或线程数。
- 使用 **MuJoCo** 进行资产可视化和独立数值对照；动力学由自研引擎求解。
- 可视化只需在推理或单 CPU env（`num_envs=1`）下实现，完整保留 MicroDuck 的可识别外观；GPU 多 env 训练无窗口运行。

## 开发环境

本机为 Apple Silicon Mac，纯 CPU 代码、本地推理和可视化可在本地运行。GPU/CUDA 代码在 SSH 远端运行，主机、目录与每次执行流程统一见 [协作与运行规则](../AGENTS.md)。
