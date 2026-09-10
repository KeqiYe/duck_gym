# MicroDuck 引擎开发路径

本文件统一记录开发顺序、阶段状态、待办与验收。稳定背景见 [项目背景](PROJECT_CONTEXT.md)，数学与接口见 [技术设计](TECHNICAL_PLAN.md)。

## 当前进度

阶段 0—3 已完成首版实现与本机验证。阶段 4 已打通 CPU PPO 训练、保存加载、推理和回放，选定 checkpoint 通过名义初态下的 30 秒与推力验收；随机初态鲁棒性和训练后期退化仍待改进，不能据此宣称通用稳定站立。阶段 5、6 已完成首版 CUDA/共享 CPU 实现、数值回归和实际 PPO 更新；阶段 7 已训练出四份方向专用策略，各自在单 CPU env 与 CUDA 上完成 2 秒起步后的连续 30 秒移动并满足当前自检门限；误差及偏航门限尚待用户确认，正式验收仍待确认。尚未验证统一策略、运行中切换方向或随机初态鲁棒性。

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

上游软约束诊断可用 `--original-softness --case microduck_original_softness`，预期可能因容差不满足而返回非零。默认结果目录按时间命名；完整参数、支持范围及参考模型区别见技术设计。阶段 1 当时仅验证 Mac；后续 Linux/CUDA 结果见阶段 5、6。

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

## 阶段 5—7 的实现与验证

- 阶段 5：`src/cuda/` 提供原生 CUDA 与可在 Mac 编译的共享 CPU 求解核心，包含 FP32/FP64、独立环境、重置、地面接触和有界电机控制。`scripts/validate_cuda.py` 进行原 CPU 对照和 CUDA 接口测试；`scripts/benchmark_cuda.py` 记录批量吞吐。隐式电机版本已通过 200 轮 FP32/FP64 回归；50 轮 FP32 满足其门限，50 轮 FP64 整机误差 2.43 mm 超过 2 mm 门限，失败记录保留。早期显式控制版本的结果不代替当前验证。
- 阶段 6：`python/duck_gym/tensor_env.py`、`scripts/train_cuda.py` 对接 rsl-rl 2.3.3；仿真状态、观测、奖励与策略在同一 CUDA 设备。`scripts/evaluate_cuda.py` 支持将权重与归一化统计加载到 CPU 本地推理。已完成 100 轮 PPO，并加载权重与归一化统计进行本地 CPU 推理和完整资产回放；恢复训练的 checkpoint 编号从 100 延续。
- 阶段 7 用户确定的目标：**前、后、左、右各 0.05 m/s，连续 30 秒**，左右为保持朝向的侧移。评估关闭自动重置，四个方向独立报告；四份方向专用策略已完成实测，结果与范围见下表；容差未确认前不标记最终验收通过。
- 阶段 7 首版自检建议（误差及偏航门限待用户确认）：允许 2 秒起步，再测量完整 30 秒；平均速度向量误差不超过 0.01 m/s、每个完整 1 秒滑动窗口误差不超过 0.02 m/s，最大偏航不超过 0.3 rad，并沿用环境的跌倒/约束失败判据。这些数值是实现中的自检建议，用于区分持续沿命令方向移动与短时位移；未经确认不得视为用户最终验收要求。速度来自真实基座位移；协议不单独约束足底滑动，不能据此证明无滑步态。
- 远端入口 `scripts/remote/run.py` 检查主机与设备，记录提交和逐文件哈希，先同步核验，再构建并以持久进程运行；各次日志、PID、退出码和 manifest 保存于约定的 `runs/<run-id>/`。

### 当前可复查结果

- `runs/gpu-20260910-213022/`：master172、物理 GPU 1（RTX 5880 Ada）、CUDA 12.8 / PyTorch 2.8.0+cu128；200 轮求解的 11 个用例在 FP32/FP64 均通过。1 秒整机对原 CPU 的 FP64 位置/旋转差为 0.0407 mm / 0.00130 rad，FP32 为 3.03 mm / 0.0211 rad；同精度共享 CPU 对照用于分离舍入差异。
- 同一目录的首轮 PPO 使用 FP32、1024 env、1 ms 物理步、20 子步/控制周期、50 轮求解预算，完成 100 轮更新、2,457,600 个控制步（49,152,000 个物理步），耗时 466.6 秒；actor 参数 L2 变化 4.9515。checkpoint 中保存优化器与归一化统计。
- `runs/stage7-checkpoint50/report.json`：四方向 CPU 评估均未跌倒，连续 32 秒（2 秒起步 + 30 秒测量），但平均速度接近零，四方向移动验收全部失败。该结果只能说明本次名义初态下学会站立。
- `runs/stage6-local-smoke/`：早期 checkpoint 的本地加载与完整资产回放验证，约 0.7 秒跌倒；不作为站立或移动通过案例。
- 本地 CTest、原有 4 个 Python 接口/渲染映射测试、新增 11 个共享核心/环境测试和 checkpoint 扩维/优化器延续测试通过。CUDA 边界测试覆盖隔离、掩码重置、快照所有权、跨 stream 调用、数值故障与重置；受控 PD/力矩饱和/外力的共享 CPU 对照由 `validate_cuda.py` 单独记录。
- `runs/gpu-20260910-214658-799256/` 的 FP32 / 50 轮基准：预热后每轮 200 个物理步、三轮计时，128 / 512 / 1024 env 分别约 31,948 / 101,619 / 155,293 个物理环境步每秒。最大锚点误差约 21.9 μm、穿透约 13.1 μm。该批量 CUDA 数据不能与单环境 CPU MuJoCo 数字直接作为等精度性能比较。
- `runs/gpu-20260910-221957-855511/` 的 `compute-sanitizer` memcheck 与 synccheck 均为零错误，随后实际完成 200 轮 PPO。接触力快照、数值故障重置与跨 stream 生命周期包含在本次检查中。
- `runs/stage7-load600/` 与 `runs/stage7-filter600/`：加入足底离地、负载转移以及低通速度奖励后，四方向名义初态仍可连续运行 32 秒，但目标方向平均速度远低于 0.05 m/s，部分方向偏航超限；全部保留为失败案例。不能用更高训练奖励替代移动验收。
- 求解预算敏感性诊断 `runs/control_iteration_audit.json`：同一段已记录动作在 50 / 100 / 200 / 400 轮预算下进行 2 秒开环回放，长期轨迹与跌倒结果不同。这不是闭环跨预算鲁棒性测试，也不证明哪个预算更准确；恢复训练默认保留求解预算，策略验收必须使用对应配置。
- `tests/test_evaluation.py` 的 3 项验收逻辑测试通过：短时匀速不能冒充 30 秒、相邻时段抵消后的平均速度不能掩盖滑动窗口超差、失败与偏航超限不能通过。早期报告使用非重叠窗口，后续报告显式标注全部滑动窗口。
- 推理批大小敏感性诊断：近临界第 800 轮策略在 CPU 的 1 / 4 env 下，初始观测相同，网络动作最大差约 `1.19e-7`，长期结果不同；固定相同动作时，原生物理跨批大小逐位一致。评估现默认逐环境执行策略网络，并记录仿真与网络批大小。该模式下，4 env 中的右移与实际单 env 的 10 秒完整刚体轨迹逐位一致；CPU/CUDA 原生固定动作跨批测试也通过。设备端缓冲轨迹改动前后，CPU 四方向 10 秒记录的所有字段逐位一致。

### 阶段 7 选定结果（2026-09-10）

四份独立策略分别使用对应方向的命令，以原始关节步态参考加 PPO 残差控制。训练为 master172 / RTX 5880 Ada、1024 env；评估为单 env，FP32、1 ms 物理步、20 子步/控制周期、50 轮求解预算。以下每项均完成 **2 秒起步 + 30 秒完整测量**，名义初态、无外推力、无跌倒或重置。左右保持初始朝向；速度为世界系实际基座位移计算。

| 方向 | 选定 checkpoint（训练 run / 轮数） | CPU 平均速度 (x,y), m/s | CUDA 平均速度 (x,y), m/s | CPU / CUDA 最大 1 秒窗口误差, m/s |
| --- | --- | --- | --- | --- |
| 前进 | `gpu-20260910-233839-702059` / 1150 | (0.04842, 0.00027) | (0.04857, 0.00059) | 0.01164 / 0.01049 |
| 后退 | `gpu-20260910-231311-223266` / 1050 | (-0.05284, -0.00110) | (-0.05268, -0.00098) | 0.01048 / 0.01152 |
| 左移 | `gpu-20260910-234015-841338` / 1150 | (0.00022, 0.04791) | (0.00006, 0.04789) | 0.01842 / 0.01800 |
| 右移 | `gpu-20260910-232454-629500` / 1150 | (-0.00083, -0.04766) | (-0.00132, -0.04797) | 0.01094 / 0.01104 |

八次评估的平均速度向量误差均不超过 0.00305 m/s，最大偏航不超过 0.2541 rad。全部满足当前建议自检协议，报告 `passed=true`；由于误差、窗口、偏航及起步容差尚待用户确认，`accepted=false`。这不代表一个统一策略能处理四方向或切换命令，也不代表无足底滑动、随机初态或跨求解预算鲁棒性。

交付目录为本地 `runs/stage7-delivery/`：`selection.json` 记录选定模型的 SHA256、原始来源与 CPU/CUDA 指标；每方向包含权重和归一化统计、原配置、原生模型/步态参考、来源清单、完整轨迹与原始报告、32 秒 MP4。原始 GPU 评估目录分别为 `gpu-20260910-235012-777003`（前）、`gpu-20260910-234059-035624`（后/右）、`gpu-20260910-235025-957240`（左）。本地 CPU 原始目录为 `stage7-forward1150`、`stage7-backward1050`、`stage7-left1150`、`stage7-right1150`。模型与视频保存在运行目录，不纳入 Git。

已有失败案例仍保留：联合策略第 1050 轮虽在 CPU 的前/左方向满足自检，CUDA 左移在 1.72 秒失败，随后才进行本表的方向专项微调；不能以本表掩盖训练与迁移敏感性。通用训练分支 `gpu-20260910-224009-924680` 在保存 1150 轮后主动停止，`termination.json` 记录实际完成更新数及原因，退出码 143 不作为自然训练完成。其余选定专项训练及最终评估正常退出；已核查本次远端无遗留训练/评估进程。

### CUDA 运行入口

```bash
# 首次在约定远端工作区安装隔离依赖；显式选择空闲 GPU。
.venv/bin/python scripts/remote/run.py --gpu 1 --setup
# 首版精度回归：FP32 / FP64，各 200 轮求解预算。
.venv/bin/python scripts/remote/run.py --gpu 1 -- python scripts/validate_cuda.py
# 实际 FP32 训练预算的单独回归。
.venv/bin/python scripts/remote/run.py --gpu 1 -- python scripts/validate_cuda.py --iterations 50 --precision fp32
# 训练入口；是否通过移动验收仍需独立检查。
.venv/bin/python scripts/remote/run.py --gpu 1 -- python scripts/train_cuda.py --task locomotion --gait --foot-clearance --num-envs 1024 --solver-iterations 50 --iterations 300
# 本地加载随模型带回的配置与参考表，单 CPU env 推理。
.venv/bin/python scripts/evaluate_cuda.py --checkpoint runs/<run-id>/train/model_<iteration>.pt --backend cpu --direction forward --output runs/local-eval
.venv/bin/python scripts/render_trajectory.py runs/local-eval/forward/trajectory.npz --follow
```

`--resume` 保留权重、优化器和归一化统计并延续迭代编号；改变动作尺度时显式指定 `--action-scale`，脚本缩放 actor 末层与探索标准差，并清理这些被重参数化变量的优化器动量。动作饱和区域无法保证与旧策略完全等价，需重新评估。`--noise-std` 是明确的探索调整，记录在新配置中。训练目录保存原生模型、元数据和步态参考的副本及哈希，避免后续生成参考表覆盖旧 checkpoint 的推理配置。
