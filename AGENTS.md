# 项目协作入口

## 开始工作

- 先阅读 `docs/PROJECT_CONTEXT.md` 了解稳定背景，再阅读 `docs/DEVELOPMENT_ROADMAP.md` 了解开发阶段、进度和待办。
- 根据当前任务检查实际文件和运行结果；上下文文档不能替代代码与验证证据。
- 项目背景已经建立；开发顺序与阶段状态以开发路径为准，实际执行范围以用户最新要求为准。

## 已确认的工程方向

- 目标为 Pollen Robotics MicroDuck；资产来源与版本见 `docs/microduck_asset_manifest.json`。
- 自研物理仿真求解器，接入 rsl-rl；明确采用原生 C++/CUDA，同时支持完整 CPU 求解。
- 多厂商 GPU 与 Kokkos 暂不作为当前实施目标；此条替代此前多厂商优先建议。
- 物理求解主线采用刚体 AVBD：最大坐标、每刚体六维局部块、增广拉格朗日关节约束，以及接触与摩擦的统一迭代；详见 `docs/ARTICULATION_SOLVER.md`。
- AVBD 的路线已确认，具体离散参数、收敛阈值和实现细节仍须推导与验证。CRBA/ABA/SAP 不作为首版必需依赖。
- 使用 Git 管理代码、配置与文档；外部资产保留来源、版本与许可信息。
- 纯 CPU 代码可以在 Mac 本地编译运行，包括 CPU 求解、测试和本地渲染测试；本机 Apple GPU 不能作为 CUDA 设备使用。
- GPU/CUDA 代码在远端运行，优先 `master172`（工作根目录 `/public/yekq6Data/codex`），连接不可用时回退 `delltower`（`/home/yekeqi/Documents/HDD1/codex`）。
- 每次远端运行前，先用 `scp` 或 `rsync` 把本次代码、配置及所需资产同步到所选主机的项目工作目录，核对完成后再在远端编译运行；不能直接运行陈旧副本或复用 Mac 二进制。
- 源码副本、编译目录和模拟结果均放在上述工作根目录下的 `duck_gym/` 内，分别使用 `source/`、`build/`、`runs/`；完整约定见 `docs/REMOTE_EXECUTION.md`。
- env 数量指并行独立仿真环境个数，配置记为 `num_envs`；环境数量、CPU/CUDA 求解后端和渲染开关独立设置。
- 可视化仅要求在推理或单 CPU env（`num_envs=1`）下实现，支持 Mac 本地查看；GPU 批量训练无窗口运行，当前不要求训练期实时渲染或远端姿态流。
- 完整保留 MicroDuck 外观，分别处理渲染网格与碰撞几何。单 CPU env 不是单核/单线程限制，设计见 `docs/TECHNICAL_PLAN.md`，验收要求统一见 `docs/DEVELOPMENT_ROADMAP.md`。

## 维护上下文

- 区分用户已确认的决策、实际检查得到的事实和待讨论的建议。
- 待定内容不是既定约束；不要自行填入机器人参数、硬件配置或训练验收数值。
- `PROJECT_CONTEXT.md` 只维护项目的大背景和长期约定，不加入阶段进度、日常运行记录、待办或讨论历史。
- 开发顺序、阶段进度、待办与验收统一维护在 `DEVELOPMENT_ROADMAP.md`；数学、接口和运行细节维护在各自技术文档。
- 用户改变长期方向时更新背景及相关技术文档；记录阶段完成状态时在开发路径附上代码或验证依据，不把计划写成已实现能力。
- 项目说明与讨论默认使用中文，代码标识符和外部项目名称保留原文。
