# MicroDuck AVBD 引擎性能

本报告比较自研刚体 AVBD 与 Newton 1.6.0 的刚体 `SolverVBD`（`rigid_compliant_alm=False`）。结果用于说明完整物理步的执行效率；接触离散、迭代停止条件和数值结果仍有差异，尚未建立等精度比较。这些数值不包含 PPO 更新，也不等于训练吞吐。

## 测量口径

- 使用官方 MicroDuck 的 15 个动态刚体、14 个铰链、11 个完整碰撞网格，仅地面接触；固定 HOME 目标，FP32，物理步长 1 ms。
- 按双方共同支持的物理模型，显式移除关节 armature、干摩擦和电机力矩上限，保留质量、惯性、几何、粘性阻尼和 HOME PD。不是完整训练执行器配置。
- 自研使用 MicroDuck 专用紧凑全局状态，最多 50 轮并可提前停止；Newton 固定 50 轮。自研每个 kernel 推进 20 个物理子步，Newton 重放包含 20 个完整物理步的 CUDA Graph。
- 每档都至少热身 **1000 个物理步且累计 10 秒 GPU 时间**；然后测量 5 次、每次 200 步，从相同初态重置，取 CUDA Event 时间中位数。排除重置、CPU 读回、PPO 与渲染；计时期间 GPU 不与其他计算进程共享。历史主表 Newton 内部缓冲验证核仍计入，其控制组差异见下文。
- 吞吐 `num_envs × 200 / median_GPU_seconds`，单位为 **物理 env-step/s**。`num_envs` 是并行独立环境数，不是每秒步数。
- 初始化包含导入、CUDA context、资产导出、模型与状态分配、编译/加载、首个 chunk 和 Graph 捕获；排除依赖安装、排队、轨迹预检和预热。首档新引擎缓存与后续缓存复用分开标注，不把缓存加载写成冷编译。

## RTX 4090

**待实测。** 2026-09-27 已确认目标为 delltower，当前该机器离线，尚未获得 4090 数字；不以其他 GPU 的结果推算替代。计划按上述口径测 512 / 1024 / 2048 / 4096 env。

## RTX 5880 Ada：已有实测

2026-09-27，RTX 5880 Ada 48 GB，驱动 570.211.01，CUDA Toolkit 12.8，PyTorch 2.8.0+cu128、Warp 1.17.0、Newton 1.6.0。原始的五次计时、初始化分解、实际热身、来源哈希和重复性记录见 [JSON 数据](benchmarks/rtx5880-avbd-2026-09-27.json)。

| env | 自研 AVBD env-step/s | Newton AVBD env-step/s | 自研 / Newton | 自研初始化 s | Newton 初始化 s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 258,120 | 95,988 | 2.69× | 117.160 | 26.802 |
| 1024 | 415,849 | 157,052 | 2.65× | 3.190 | 13.331 |
| 2048 | 401,557 | 230,386 | 1.74× | 3.202 | 15.731 |
| 4096 | 420,920 | 263,993 | 1.59× | 3.272 | 21.972 |

512 env 为该 run 的新引擎缓存，后续规模复用；自研首档初始化约 117 s 中，约 114 s 是编译/加载。4096 env 的约 3.27 s 是缓存复用初始化，不能与冷编译混用。

主表的历史 Newton 适配器还包含官方可选缓冲诊断核和全零关节力映射。关闭这两项额外工作的控制组，在 512 / 4096 env 分别达到 **96,430 / 264,566 env-step/s**；再启用接触历史时为 **95,838 / 268,378 env-step/s**。4096 env 下，自研仍为两个控制组中较快者的约 **1.57×**。

当前适配器默认去掉额外工作，Newton 接触历史默认关闭。4090 的新测量应使用当前实现；与本表历史主测跨卡比较时，代码和软件环境差异也必须同时考虑，不能把变化全部解释为 GPU 型号差异。

## 实现如何减少开销

- 每个环境在一个 CUDA warp 中完成刚体求解，同一 kernel 内循环多个物理子步，减少步进调度和中间状态搬运。
- 针对 MicroDuck 固定拓扑压缩状态容量，并通过 CSR 邻接只访问刚体相关的关节约束。
- 复用关节角、有效接触与局部刚体块，避免为每个局部更新遍历全部模型。

这些是代码层面的实现差异，不是经过逐项消融证明的提速百分比。Newton 的通用实现、接触流形和迭代策略也不同；目前没有把总倍数分摊给某一项优化。实现入口见 [CUDA 绑定](../src/cuda/extension.cu)、[共享 AVBD 核心](../src/cuda/kernel.cuh) 和 [技术设计](TECHNICAL_PLAN.md#16-microduck-专用后端与性能实验接口)。

## 数值结果与结论边界

短轨迹的执行检查包括完整逐步预检、有限状态、已暴露的容量计数、初态重置和重复运行。Newton 默认并行归约存在非确定性：严格重复性检查未通过的差异保留在数据中，并未通过放宽数值门限改成通过。自研在 200 步轨迹中有 22 步达到迭代上限，不能把相同轮数预算解释为相同求解质量。

延长到 10 秒、8 env、逐 1 ms 记录全部 15 个动态刚体时，相对 Newton 的根部最大位置相对差为 **6.61%**、全身最大位置相对差为 **6.75%**、全身最大高度相对差为 **22.05%**；全身最大绝对位置差为 **15.04 mm**。该固定 HOME 工况两边都发生倾倒，峰值包含倾倒过程。位置百分比以 Newton 世界位置向量范数为分母，高度百分比以 Newton 高度绝对值为分母；Newton 是参考实现，不是物理真值。详细范围见 [长轨迹记录](DEVELOPMENT_ROADMAP.md#连续-10-秒的跨引擎误差)。

因此，本报告支持**指定共同模型和固定 HOME 工况下的纯引擎效率**，不支持等精度优势、稳定行走/空翻成功或完整 PPO 训练速度的推断。

## 复现

依赖版本在 [Newton 基准依赖](../requirements-benchmark-newton.txt)。按 [远端运行规则](../AGENTS.md) 检查设备、同步源码并构建；以下命令要求 delltower 的 4090 已上线且 GPU 0 空闲，基准隔离环境已准备：

```bash
.venv/bin/python scripts/remote/run.py \
  --host delltower --gpu 0 \
  --python-env venv-benchmark-newton --build-config cuda-avbd-common \
  -- python scripts/benchmark_engines.py \
  --engines avbd newton-avbd --physics-profile avbd-common \
  --envs 512 1024 2048 4096 --avbd-specialized \
  --dt 0.001 --avbd-iterations 50 --steps 200 --chunk-steps 20 \
  --warmup-steps 1000 --warmup-seconds 10 --repetitions 5 \
  --nconmax 32 --njmax 128 --newton-avbd-replay-policy report
```

`report` 仅如实保留 Newton 非确定性复现差异，不豁免非有限状态、已暴露溢出或初态重置检查。实际设备编号必须通过当次 `nvidia-smi` 确认。新运行的 manifest、源码快照、日志、逐步预检和实际二进制保存在对应 `runs/<run-id>/`；公开 JSON 是可审阅的计时摘录，完整本地产物不随 Git clone 传递。
