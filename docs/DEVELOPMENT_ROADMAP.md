# MicroDuck 引擎开发路径

本文件统一记录开发顺序、阶段状态、待办与验收。稳定背景见 [项目背景](PROJECT_CONTEXT.md)，数学与接口见 [技术设计](TECHNICAL_PLAN.md)。

## 当前进度

阶段 0 的安装、资产导入和完整离屏显示已完成；阶段 1 的 CPU 首版与限定场景对照已完成。当前停在阶段 1 验收结果，阶段 2 尚未开始。

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
