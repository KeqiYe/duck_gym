# MicroDuck 引擎开发路径

本文件统一记录开发顺序、阶段状态、待办与验收。稳定背景见 [项目背景](PROJECT_CONTEXT.md)，数学与接口见 [技术设计](TECHNICAL_PLAN.md)。

## 当前进度

阶段 0—3 已完成首版实现与本机验证。阶段 4 已打通 CPU PPO 训练、保存加载、推理和回放，选定 checkpoint 通过名义初态下的 30 秒与推力验收；随机初态鲁棒性和训练后期退化仍待改进，不能据此宣称通用稳定站立。阶段 5 尚未开始。

## 开发顺序

| 阶段 | 内容 |
| --- | --- |
| 0 | 整理依赖并编写安装脚本，使用 MuJoCo 完整可视化机器人资产。 |
| 1 | 建立 CPU 动力学模拟引擎，与 MuJoCo 对比计算准确性和吞吐。 |
| 2 | 通过 pybind 将 CPU 模拟引擎接入 rsl-rl。 |
| 3 | 实现动力学模拟状态到可视化的数据传递。 |
| 4 | 通过 CPU 引擎训练稳定站立。 |
| 5 | 实现 CUDA 动力学引擎。 |
| 6 | 将 CUDA 引擎接入 rsl-rl。 |
| 7 | 训练前进、后退和左右移动。 |

## 推进与验收

- 按上述顺序推进，先完成 CPU 训练闭环，再开发 CUDA。
- 每阶段开展时确定具体测试条件、时间步与容差，不预设未经验证的性能数值。
- 阶段完成时在本文附上代码位置、运行命令和验证结果；设计不代表功能已实现。
- 实机部署、复杂地形及通用机器人扩展另行确定范围。

## 阶段 0、1 的交付与验证（2026-09-10）

### 阶段 0

- `scripts/setup.sh`：创建项目虚拟环境、安装锁定依赖、下载校验资产、构建 CPU 引擎并运行 CTest；要求 Python 3.11+、CMake 3.20+、C++17 编译器。
- `scripts/fetch_assets.py`：逐文件下载与重试，校验 38 个网格、模型及上游说明/许可共 41 个文件。资产保留在 Git 忽略的 `assets/microduck/`，来源集中在资产清单。
- `scripts/view_assets.py`：完整模型的正面、侧面、背面图，使用 38 个网格、38 个材质、15 个刚体和 14 个 hinge；显示时不推进物理。
- 本机会话当前没有活动显示器，MuJoCo/GLFW 原生窗口不能使用。查看器检测后自动回退离屏图；离屏渲染和回退路径均已验证，有活动显示器时的交互窗口仍待实测。

### 阶段 1

- `duck_cpu` 静态库与 `duck_sim` 命令行程序：刚体 AVBD、hinge/限位、阻尼/armature/干摩擦、地面接触与库仑摩擦、状态重置、CPU 独立多环境。
- CTest 的旋转/矩阵不变量、hinge Jacobian 数值差分、重置与环境隔离测试通过；UBSan 测试及整机短运行通过。当前 macOS 的 ASan 在运行时初始化阶段死锁，未作为通过项，相关进程已停止。
- `scripts/validate_cpu.py` 的 11 个对照用例通过所列容差：自由落体、单摆、电机、干摩擦/静摩擦、限位、球落地、盒静止/滑动、自由旋转、MicroDuck 1 秒被动释放。

整机主要结果（CPU FP64、`num_envs=1`、每步最多 200 轮）：

| 对照条件 | 最大质心位置差 | 最大旋转差 | 结论 |
| --- | --- | --- | --- |
| 较硬 MuJoCo 参考，1 ms，1 秒 | 3.52 mm | 0.0372 rad | 满足 20 mm / 0.05 rad 容差 |
| 较硬 MuJoCo 参考，0.5 ms，1 秒 | 1.21 mm | 0.0149 rad | 缩小时间步后误差下降 |
| 上游软约束参考，1 ms，1 秒 | 13.2 mm | 0.0699 rad | 旋转差超过上述容差，保留为模型差异诊断 |

1 ms 整机测试中，最大关节锚点误差约 7 μm，最大穿透约 3.5 μm。部分步未达到更严格的内部停止条件，报告保留 `unconverged_steps`，不能把误差验收通过等同于每步完全收敛。

本机一次三轮中位数基准约为自研 911、MuJoCo 54,141 步/秒。计时排除加载、渲染及轨迹写盘，双方执行各自原生物理步进；这是当前配置的基线，不是相同收敛精度下的性能优劣证明。完整输出位于本地 `runs/stage01-final/report.json`，包含命令、源码/二进制哈希、环境版本、误差和每轮计时；原始软约束与缩小时间步实验分别在 `runs/stage1-microduck-long/`、`runs/stage1-refinement/`。

### 重现命令

在仓库根目录执行：

```bash
bash scripts/setup.sh
.venv/bin/python scripts/view_assets.py
.venv/bin/mjpython scripts/view_assets.py --interactive
.venv/bin/python scripts/validate_cpu.py --microduck
.venv/bin/python scripts/validate_cpu.py --microduck --case microduck_release --dt 0.0005
```

上游软约束诊断可用 `--original-softness --case microduck_original_softness`，预期可能因容差不满足而返回非零。默认结果目录按时间命名；完整参数、支持范围及参考模型区别见技术设计。Linux 与 GPU 运行尚未验证。

## 阶段 2—4 的实现与验收

### 阶段 2：CPU / pybind / rsl-rl

`src/bindings.cpp`、`python/duck_gym/env.py` 和 `scripts/train_standing.py` 提供原生 CPU 批量步进、14 维动作、52 维观测及真实 PPO 更新。依赖版本及入口分别见 `requirements-training.txt`、`scripts/setup_training.sh`。本机验证了环境隔离、串行/多线程一致性、数组所有权、单环境超时重置、外力单位，以及不依赖 MuJoCo 步进的训练路径；阶段 1 的 11 个物理对照用例和 CTest/UBSan 回归均通过。

### 阶段 3：完整资产回放

`scripts/render_trajectory.py` 将自研引擎轨迹渲染为本地 MP4 和 PNG，保留 38 个网格和 38 个材质。位姿映射通过数值测试，包括独立移动一个连杆时不影响其他连杆，也不掩盖最大坐标的约束残差。本机已验证 30 秒轨迹的离屏回放；训练不渲染，单 CPU env 推理完成后在本地观看。

### 阶段 4：站立

用户确认首版验收为 **连续站立 30 秒，并通过小幅推力干扰测试**。当前采用的具体工程协议：在第 5、10、15、20 秒分别沿世界系 `+x/-x/+y/-y` 向基座质心施加机器人重力的 5%，持续 0.2 秒；按模型总质量折合约 0.362 N。评估关闭自动重置。失败判据为倾斜超过 0.5 rad、基座高度偏差超过 40 mm、关节误差或穿透超过 5 mm，最终水平位移最大值还须小于 50 mm。这些容差是本次仿真验收协议，不是硬件安全指标。

### 本次训练与限定验收结果（2026-09-10）

Mac CPU FP64、8 env / 8 工作线程，PPO 训练 300 轮，共 57,600 个环境控制步（1,152,000 个物理步），约 478 秒。最终 actor 参数相对初始化的 L2 变化为 1.6804，实际进行了梯度更新；checkpoint 中观测归一化样本数为 57,600，恢复统计与确定性推理已检查。另以短训练验证优化器恢复及继续训练的 checkpoint 编号。

本次选用 **`runs/stage4-ppo-residual/model_150.pt`**（完成 151 轮更新）进行演示。不能默认最后一个 checkpoint 最好，也不能把下表的受限通过表述为随机初态鲁棒性通过。

| 条件 | 连续时长 | 最大倾斜 | 结果 |
| --- | --- | --- | --- |
| 固定 PD 基线、名义初态、四方向推力 | 30 s | 0.2665 rad | 通过，没有 PPO 策略 |
| 第 150 轮模型、名义初态、四方向推力 | 30 s | 0.2840 rad | 通过；最大水平位移 19.17 mm |
| 第 299 轮模型、名义初态、四方向推力 | 29.52 s | 超过 0.5 rad | 失败，训练后期退化 |
| 第 150 轮模型、随机初态 seed=456 | 0.38 s | 超过 0.5 rad | 失败，尚未到推力时刻 |

选定模型的最大关节锚点误差约 1.49 μm，最大穿透约 4.26 μm；1,499/1,500 个控制周期包含至少一个耗尽迭代预算的子步。误差在本次容差内，不能视为每步完全收敛。当前 PPO 也没有证明优于固定 PD 基线。后续应优先定位初态敏感性、控制与积分的长期稳定性，并改善策略选择和训练稳定性。

通过案例的原始轨迹、报告、30 秒 MP4 位于 `runs/stage4-residual150/`；失败案例保留在 `runs/stage4-final/`、`runs/stage4-final-random/`、`runs/stage4-accepted-random/`。选择记录位于训练目录的 `selected_checkpoint.json`，记录模型哈希与验收报告。所有结果均来自自研 CPU 引擎，MuJoCo 没有推进训练或推理物理。

### 重现训练与回放

```bash
bash scripts/setup_training.sh
.venv/bin/python scripts/train_standing.py --output runs/standing --num-envs 8 --threads 8 --iterations 300
.venv/bin/python scripts/evaluate_standing.py --checkpoint runs/standing/model_150.pt --output runs/standing-eval
.venv/bin/python scripts/render_trajectory.py runs/standing-eval/trajectory.npz
```

上述 checkpoint 编号对应本次已验证配置；其他训练应检查各 checkpoint 的评估结果，不能将末轮模型自动当作验收通过。

每次训练目录包含 `config.json`、PPO checkpoint（含归一化统计）、TensorBoard 日志和 `training_report.json`；每次评估保存 `report.json` 与逐控制周期全部连杆位姿/实际推力的 `trajectory.npz`。视频、网格、构建和模型输出不纳入 Git，源码和配置纳入 Git。CPU 结果不能替代 CUDA 或实机验证；当前仍仅支持地面接触，无自碰撞或真实 BAM 执行器。
