# CPDOT Python Reproduction

[中文](README.md) | [English](README.en.md)

CPDOT Python Reproduction 是对论文 **Multi-Nonholonomic Robot Object
Transportation with Obstacle Crossing using a Deformable Sheet** 中核心规划
流程的 Python 复现。项目包含算法实现、测试、可视化脚本和从 C++ 输出提取的
fixture 数据，不包含原 C++/ROS 工程源码和 catkin 编译产物。

默认配置为 3 车编队，支持 `N in [3, 7]`。论文流程相关实验建议使用
`--mode source --source-solver-method ipopt`。

## Visual Results

当前 `outputs/` 下的 source 动画结果。

### 3 vehicles

<p>
  <img src="outputs/cpdot_source_animation.gif" alt="CPDOT source animation with 3 vehicles" width="480">
  <img src="outputs/cpdot_source_animation_002.gif" alt="CPDOT source animation with 3 vehicles passing obstacles from different sides" width="480">
</p>

### 5 vehicles

<p>
  <img src="outputs/cpdot_source_animation_001.gif" alt="CPDOT source animation with 5 vehicles" width="480">
</p>

## Installation

需要 Python 3.10。推荐使用 conda：

```bash
cd CPDOT

conda env create -f environment.yml
conda activate cpdot-py
```

已有环境可更新：

```bash
conda env update -f environment.yml --prune
conda activate cpdot-py
```

也可以使用 pip：

```bash
python -m pip install -r requirements.txt
```

CasADi 的 pip 包自带 IPOPT 二进制后端，通常不需要单独编译 IPOPT。

## Full Test Commands

以下命令覆盖测试收集、全量测试、快速 demo、source stage pipeline、完整 source
流程、动画生成和动画播放。

```bash
cd CPDOT
conda activate cpdot-py

# 1. 确认测试可收集
python -m pytest --collect-only tests -q

# 2. 全量测试
python -m pytest tests -q

# 3. 清空旧输出，确保后续文件名固定
rm -rf outputs

# 4. 快速 smoke demo，生成静态图
python main.py --mode fast --scene-seed 0

# 5. 快速 smoke demo，生成动画 GIF
python main.py --mode fast --scene-seed 0 --animate

# 6. source stage 逐段运行并保存中间结果
python main.py --mode source --scene-seed 0 --source-stage topo
python main.py --mode source --scene-seed 0 --source-stage combo
python main.py --mode source --scene-seed 0 --source-stage corridor
python main.py --mode source --scene-seed 0 --source-stage coarse
python main.py --mode source --scene-seed 0 --source-stage plan \
  --source-warm-starts 1 --source-initial-warm-starts 1 \
  --source-solver-method ipopt

# 7. 可视化 plan stage 结果
python scripts/visualize_stage.py \
  --npz outputs/source_stage_plan.npz \
  --output outputs/source_stage_plan.png

# 8. 论文流程完整链路，生成静态图和动画 GIF
python main.py --mode source --scene-seed 0 \
  --source-warm-starts 1 --source-initial-warm-starts 1 \
  --source-solver-method ipopt --animate

# 9. 播放动画
xdg-open outputs/cpdot_animation.gif
xdg-open outputs/cpdot_source_animation.gif
```

可视化结果（.png/.gif/.npz）保存在 `outputs/`，若无则会在首次运行时自动生成。

## CLI Modes

| Mode | Command | Description |
|---|---|---|
| `fast` | `python main.py --mode fast` | 快速 smoke demo，使用轻量启发式平滑器，不作为论文复现实验数据 |
| `source` | `python main.py --mode source --source-solver-method ipopt` | 运行论文核心流程：TopologyPRM、同伦组合、走廊、粗规划、Plan_fm |
| `source-single` | `python main.py --mode source-single` | 单机器人 Plan、diff-drive、car-like replan 分支诊断 |

`outputs/` 中的输出文件不会覆盖已有结果；若目标文件存在，会自动追加
`_001`、`_002` 等后缀。需要固定文件名时，先执行 `rm -rf outputs`。

## 目录结构

```
.
├── README.md
├── README.en.md
├── environment.yml / requirements.txt
├── main.py                        # CLI 入口
├── cpdot_py/                      # 算法包
│   ├── topo_prm.py                  # Topological PRM(guard/connector)
│   ├── homotopy.py                  # 同伦类组合 + 评分 + 走廊
│   ├── coarse_path_planner.py       # Hybrid A* + 2D DP heuristic
│   ├── sfc.py                       # DecompROS ellipsoid 安全飞行走廊
│   ├── forward_kinematics.py        # 柔性 sheet taut-subset + KKT
│   ├── optimizer.py                 # 4 个 NLP(scipy 后端)
│   ├── optimizer_casadi.py          # 4 个 NLP(CasADi/IPOPT 后端)
│   ├── formation.py                 # FormationPlanner / Plan_fm 主循环
│   ├── env.py / geometry.py         # 障碍 / 几何
│   ├── states.py                    # TrajectoryPoint / FullStates / Constraints
│   ├── metrics.py                   # ring-Laplacian formation_similarity
│   ├── cpp_fixtures.py              # 加载 cpp_fixtures/ 下的 YAML
│   └── visualization.py             # 静态图 / 动画
├── tests/                         # pytest 测试
├── scripts/                       # 批量实验、stage 可视化、Gazebo 比对脚本
├── cpp_fixtures/                  # 论文原版 C++ NLP 输出(N=3 + N=5)
│   └── flexible_formation/
│       ├── 3/                       # traj_3R1000.yaml + traj_real3R.yaml + ...
│       └── 5/
└── outputs/                       # demo 输出目录，运行时自动生成
```

## 算法 ↔ 代码对照

| 算法步骤 | Python 模块 |
|---|---|
| 中心拓扑 PRM | `cpdot_py.TopologyPRM` |
| 同伦类组合枚举 + safety / length / homotopy 评分 | `cpdot_py.cal_combination` |
| 每机器人 bbox 半空间走廊 | `cpdot_py.cal_corridors` |
| Hybrid A* 粗规划 | `cpdot_py.CoarsePathPlanner` |
| DecompROS 安全飞行走廊 | `cpdot_py.generate_sfc` |
| 柔性 sheet 正运动学(taut-subset + KKT) | `cpdot_py.ForwardKinematics` |
| 4 NLP(car-like / diff-drive / replan / formation) | `cpdot_py.{CarLike,DiffDrive,CarLikeReplan,Formation}NLPProblem` |
| Plan_fm warm-start 主循环 | `cpdot_py.FormationPlanner.plan_fm_from_guess` |

## Notes

- **求解器后端**:scipy(L-BFGS-B + 有限差分,默认)或 CasADi+IPOPT
  (`--source-solver-method ipopt`)。scipy 仅适合 smoke / `--mode fast`,
  论文数据用 IPOPT。
- **C++ 原码 bug 复现开关**:论文 C++ 源码里有三个已知 bug
  (`IdentifyHomotopy.cpp` 的 `BeyondInterdisCons` 双索引和无条件 `return true`、
  `combinations[sorted_indices[sorted_indices[i]]]` 双重索引、`Plan_fm` 在
  `warm_start == 5` 的早返回)。Python 默认走作者本意修正版;
  `--source-strict-homotopy-bugs` 和 `--source-strict-cpp-early-return` 复现源码 bug。
- **fixture 数据**:`cpp_fixtures/` 来自原 C++ 仓库
  `formation_planner/traj_result/flexible_formation/{3,5}/` 输出，用作 Python
  端对照基准。
- **回归基准**:Python 端 IPOPT 在 N=3 fixture 场景下与 C++ NLP 输出的 `tf`
  偏差由 `tests/test_cpp_baseline_diff.py` 约束在 0.2% 以内。
