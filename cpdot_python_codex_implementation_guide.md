# Codex 任务说明：用 Python 复现 CPDOT 核心逻辑并在自定义场景中测试

## 0. 项目目标

本项目目标不是完整重写 CPDOT 的 ROS/Gazebo/C++ 工程，而是用 Python 实现其核心算法思想的简化版，用于在我们自己的二维场景中测试。

核心目标：

1. 构建二维障碍物环境。
2. 实现多机器人编队路径规划。
3. 实现拓扑候选路径搜索，至少能生成多条可行绕障路径。
4. 实现编队约束，包括机器人间距、队形保持、柔性 sheet 的几何拉伸约束。
5. 实现轨迹平滑与避障优化。
6. 提供可视化动画与实验指标。

最终交付一个可以运行的 Python demo：

```bash
python main.py
```

运行后应展示：

- 2D 地图；
- 障碍物；
- 若干个机器人；
- 起点和终点；
- 规划路径；
- 编队运动动画；
- 避障与队形保持效果；
- 基础实验指标。

---

## 1. 总体实现原则

请遵循以下原则：

1. 先做最小可运行版本，再逐步增强。
2. 每完成一个阶段，都必须写自测脚本或运行检查。
3. 不要一开始引入 ROS、Gazebo、MOSEK 或复杂非线性优化器。
4. 优先保证代码结构清晰、模块边界明确、结果可视化。
5. 所有关键函数都应有 docstring。
6. 所有重要中间结果都应可画图检查。
7. 每一步完成后，更新 `README.md` 的当前进度与运行方式。
8. 不要把所有逻辑写进一个大文件。

---

## 2. 推荐项目结构

请创建如下结构：

```text
cpdot_py/
  README.md
  requirements.txt
  main.py

  config/
    scene_01.yaml

  cpdot_py/
    __init__.py

    env/
      __init__.py
      map2d.py
      obstacle.py
      collision.py

    planner/
      __init__.py
      prm.py
      astar.py
      path_utils.py

    formation/
      __init__.py
      formation.py
      sheet.py

    optimization/
      __init__.py
      trajectory_opt.py
      cost_functions.py

    sim/
      __init__.py
      simulator.py
      robot_model.py

    vis/
      __init__.py
      plot.py
      animation.py

    metrics/
      __init__.py
      evaluator.py

  tests/
    test_collision.py
    test_prm.py
    test_formation.py
    test_trajectory_opt.py
```

---

## 3. 依赖要求

请优先使用常见 Python 科学计算库：

```text
numpy
scipy
matplotlib
networkx
pyyaml
shapely
```

可选增强：

```text
casadi
```

但第一版不要依赖 CasADi。第一版轨迹优化优先用 `scipy.optimize.minimize`。

请生成 `requirements.txt`。

---

## 4. 分阶段任务

# Stage 1：二维环境与可视化

## 目标

实现一个简单二维场景，包括：

- 地图边界；
- 圆形障碍物；
- 矩形障碍物；
- 起点；
- 终点；
- 绘图函数。

## 需要实现的模块

### `cpdot_py/env/obstacle.py`

实现：

- `CircleObstacle`
- `RectangleObstacle`
- 每个障碍物提供：
  - `contains(point)`
  - `distance(point)`
  - `intersects_segment(p1, p2)`

### `cpdot_py/env/map2d.py`

实现：

- `Map2D`
- 保存地图尺寸、障碍物列表、起点、终点。
- 提供：
  - `is_in_bounds(point)`
  - `is_collision(point, clearance=0.0)`
  - `segment_is_collision_free(p1, p2, clearance=0.0)`

### `cpdot_py/vis/plot.py`

实现：

- `plot_map(map2d, ax=None)`
- 能画出障碍物、起点、终点、边界。

## 自测要求

完成后运行：

```bash
python main.py --stage env
```

应显示一个 2D 地图窗口。

## Stage 1 自我审查清单

- [ ] 地图边界正确显示。
- [ ] 圆形障碍物正确显示。
- [ ] 矩形障碍物正确显示。
- [ ] 起点和终点正确显示。
- [ ] 点碰撞检测可用。
- [ ] 线段碰撞检测可用。
- [ ] `README.md` 已更新运行方法。

---

# Stage 2：PRM 路径搜索

## 目标

实现一个基础 PRM planner，用来在障碍物环境中生成从起点到终点的路径。

## 需要实现的模块

### `cpdot_py/planner/prm.py`

实现：

- 随机采样自由空间点；
- 将起点和终点加入图；
- 使用 k-nearest 或 radius 方式连边；
- 连边前检查线段是否碰撞；
- 输出 graph。

核心类建议：

```python
class PRMPlanner:
    def __init__(self, map2d, num_samples=500, k_neighbors=10, clearance=0.2, seed=0):
        ...

    def build_graph(self):
        ...

    def plan(self, start, goal):
        ...
```

### `cpdot_py/planner/astar.py`

实现 A* 或直接调用 networkx shortest path。

### `cpdot_py/planner/path_utils.py`

实现：

- 路径长度计算；
- 路径插值；
- 路径平滑。

## 自测要求

运行：

```bash
python main.py --stage prm
```

应显示：

- PRM 采样点；
- PRM 边；
- 起点到终点的路径。

## Stage 2 自我审查清单

- [ ] PRM 图能成功生成。
- [ ] 路径不会穿过障碍物。
- [ ] 改变随机种子能生成不同采样图。
- [ ] 如果路径失败，程序能给出明确提示，而不是崩溃。
- [ ] `README.md` 已更新运行方法。

---

# Stage 3：多条候选路径与简化拓扑搜索

## 目标

CPDOT 的核心之一是寻找拓扑不同的候选路径。Python 简化版不必严格实现完整同伦判别，但至少要能生成多条几何上明显不同的候选路径。

## 实现方式建议

使用以下任一方法：

1. 在 PRM 图中生成 k-shortest paths。
2. 对已找到路径上的边增加惩罚，再重复搜索。
3. 对不同障碍物左右侧构造路径引导点。

推荐先实现方法 1 + 2。

## 需要实现

在 `PRMPlanner` 中添加：

```python
def plan_k_paths(self, start, goal, k=5):
    ...
```

返回多条候选路径，按路径长度排序。

## 可视化要求

运行：

```bash
python main.py --stage kpaths
```

应显示多条不同颜色的候选路径。

## Stage 3 自我审查清单

- [ ] 至少能生成 2 条候选路径。
- [ ] 每条路径都无碰撞。
- [ ] 路径长度有统计输出。
- [ ] 多条路径在图上能区分。
- [ ] 如果只找到一条路径，程序应说明原因。

---

# Stage 4：编队模型与柔性 sheet 几何约束

## 目标

实现 5 个机器人组成的编队。Leader 沿路径前进，followers 根据相对 offset 跟随。

初始版本采用几何队形，不做真实物理仿真。

## 需要实现的模块

### `cpdot_py/formation/formation.py`

实现：

```python
class Formation:
    def __init__(self, offsets):
        ...

    def get_robot_positions(self, leader_pose):
        ...

    def formation_error(self, positions):
        ...
```

默认五机器人队形建议：

```text
robot 1: leader / front center
robot 2: left front
robot 3: right front
robot 4: left rear
robot 5: right rear
```

也可以使用一个近似五点 sheet 支撑结构。

### `cpdot_py/formation/sheet.py`

实现柔性 sheet 的简化约束：

- 相邻机器人最大距离；
- 相邻机器人最小距离；
- 编队整体宽度限制；
- 编队与障碍物的 clearance 检查。

核心函数：

```python
def sheet_constraint_violation(positions, edges, min_dist, max_dist):
    ...
```

## 自测要求

运行：

```bash
python main.py --stage formation
```

应显示：

- leader 路径；
- 5 个机器人沿路径移动；
- 机器人之间连线形成 sheet；
- 动画或多帧轨迹图。

## Stage 4 自我审查清单

- [ ] 5 个机器人位置计算正确。
- [ ] 机器人之间的连接关系显示正确。
- [ ] 编队不会明显变形到不合理状态。
- [ ] 可计算队形误差。
- [ ] 可计算 sheet 约束违反量。

---

# Stage 5：编队可行性检查与路径选择

## 目标

对 Stage 3 生成的多条候选路径进行评估，选择最适合编队通过的路径。

评估指标包括：

1. leader 路径长度；
2. 所有机器人是否碰撞；
3. sheet 是否过度拉伸；
4. 编队整体是否与障碍物冲突；
5. 路径平滑程度。

## 需要实现

在 `formation/formation.py` 或新建 `planner/formation_path_selector.py` 中实现：

```python
def evaluate_formation_path(path, formation, map2d):
    ...


def select_best_path(candidate_paths, formation, map2d):
    ...
```

## 自测要求

运行：

```bash
python main.py --stage select
```

应输出每条候选路径的指标，例如：

```text
Path 0: length=12.4, collision=False, sheet_violation=0.03, smoothness=1.2, score=14.1
Path 1: length=14.8, collision=False, sheet_violation=0.00, smoothness=0.7, score=15.5
Selected path: 0
```

并画出最佳路径。

## Stage 5 自我审查清单

- [ ] 每条候选路径都有评分。
- [ ] 有碰撞的路径不会被选中。
- [ ] sheet 违反严重的路径不会被选中。
- [ ] 输出的最佳路径在图上高亮。
- [ ] 指标解释写入 README。

---

# Stage 6：轨迹平滑与优化

## 目标

将离散路径优化为更平滑、对编队更友好的轨迹。

第一版使用 `scipy.optimize.minimize`。

## 优化变量

可以将路径中间点作为优化变量：

```text
p_1, p_2, ..., p_n
```

起点和终点固定。

## 代价函数建议

总代价：

```text
J = w_len * path_length
  + w_smooth * smoothness
  + w_obs * obstacle_penalty
  + w_form * formation_penalty
  + w_sheet * sheet_penalty
```

其中：

- `path_length`：路径长度；
- `smoothness`：二阶差分平滑项；
- `obstacle_penalty`：距离障碍物太近时惩罚；
- `formation_penalty`：编队偏离期望队形的惩罚；
- `sheet_penalty`：相邻机器人距离超过阈值的惩罚。

## 需要实现模块

### `optimization/cost_functions.py`

实现各个代价项。

### `optimization/trajectory_opt.py`

实现：

```python
class TrajectoryOptimizer:
    def __init__(self, map2d, formation, weights):
        ...

    def optimize(self, path):
        ...
```

## 自测要求

运行：

```bash
python main.py --stage optimize
```

应显示：

- 优化前路径；
- 优化后路径；
- 代价下降曲线或优化前后指标对比。

## Stage 6 自我审查清单

- [ ] 优化后路径仍然从起点到终点。
- [ ] 优化后路径不穿障碍物。
- [ ] 路径比优化前更平滑。
- [ ] 若优化失败，程序保留原路径并输出原因。
- [ ] 所有代价项可单独打印检查。

---

# Stage 7：简化非完整机器人模型

## 目标

让轨迹不只是几何路径，而是更接近机器人可执行的轨迹。

第一版使用 unicycle 模型：

```text
x_dot = v cos(theta)
y_dot = v sin(theta)
theta_dot = omega
```

## 需要实现模块

### `sim/robot_model.py`

实现：

```python
class UnicycleModel:
    def step(self, state, control, dt):
        ...
```

### `sim/simulator.py`

实现一个简单路径跟踪器：

- pure pursuit 或 waypoint following；
- 限制 `v_max` 和 `omega_max`；
- 输出每个机器人的状态轨迹。

## 自测要求

运行：

```bash
python main.py --stage sim
```

应显示：

- 几何规划路径；
- 实际跟踪轨迹；
- 机器人朝向。

## Stage 7 自我审查清单

- [ ] 每个机器人轨迹连续。
- [ ] 速度不超过限制。
- [ ] 角速度不超过限制。
- [ ] 不出现明显瞬移。
- [ ] 动画中能看到机器人沿路径运动。

---

# Stage 8：实验指标与报告输出

## 目标

生成可用于向师姐汇报的结果。

## 指标

实现以下指标：

1. `success`：是否到达终点；
2. `collision_count`：碰撞次数；
3. `min_obstacle_distance`：最小障碍物距离；
4. `path_length`：路径长度；
5. `formation_error_mean`：平均队形误差；
6. `formation_error_max`：最大队形误差；
7. `sheet_violation_max`：最大 sheet 约束违反量；
8. `planning_time`：规划耗时。

## 需要实现模块

### `metrics/evaluator.py`

实现：

```python
def evaluate_run(result):
    ...
```

## 输出要求

运行：

```bash
python main.py --stage full
```

应输出：

```text
=== Experiment Result ===
Success: True
Planning time: 0.83 s
Path length: 18.42
Collision count: 0
Min obstacle distance: 0.31
Formation error mean: 0.08
Formation error max: 0.24
Sheet violation max: 0.00
```

并保存：

```text
outputs/scene_01_path.png
outputs/scene_01_animation.gif
outputs/scene_01_metrics.json
```

## Stage 8 自我审查清单

- [ ] 完整 demo 一条命令可运行。
- [ ] 图片能保存。
- [ ] 指标 JSON 能保存。
- [ ] README 中有结果截图说明。
- [ ] 至少有一个成功场景。

---

## 5. 配置文件格式

请使用 YAML 管理场景，例如：

```yaml
map:
  width: 20.0
  height: 12.0

start: [1.0, 6.0]
goal: [18.0, 6.0]

obstacles:
  - type: circle
    center: [7.0, 6.0]
    radius: 1.2
  - type: rectangle
    center: [12.0, 5.5]
    size: [2.0, 4.0]

formation:
  offsets:
    - [0.0, 0.0]
    - [-0.8, 0.6]
    - [-0.8, -0.6]
    - [-1.6, 0.6]
    - [-1.6, -0.6]
  sheet_edges:
    - [0, 1]
    - [0, 2]
    - [1, 3]
    - [2, 4]
    - [1, 2]
    - [3, 4]

planner:
  num_samples: 800
  k_neighbors: 12
  clearance: 0.25
  seed: 0

optimizer:
  weights:
    length: 1.0
    smooth: 5.0
    obstacle: 20.0
    formation: 5.0
    sheet: 10.0
```

---

## 6. `main.py` 命令行接口要求

请支持：

```bash
python main.py --stage env
python main.py --stage prm
python main.py --stage kpaths
python main.py --stage formation
python main.py --stage select
python main.py --stage optimize
python main.py --stage sim
python main.py --stage full
```

同时支持：

```bash
python main.py --config config/scene_01.yaml --stage full
```

---

## 7. README 要求

每完成一个阶段，都更新 README。README 至少包括：

1. 项目目标；
2. 安装方式；
3. 运行方式；
4. 各阶段说明；
5. 当前完成情况；
6. 示例图；
7. 与 CPDOT 原始工程的关系；
8. 当前简化假设；
9. 后续计划。

需要明确写出：

```text
本项目是 CPDOT 核心逻辑的 Python 简化复现，不是 ROS/Gazebo/C++ 工程的完整复刻。
```

---

## 8. 自我审查与更新机制

每完成一个 stage，请执行以下流程：

1. 运行对应 stage 命令。
2. 检查是否有异常报错。
3. 检查图像结果是否合理。
4. 运行相关 tests。
5. 更新 README 的进度。
6. 写一段简短开发日志到 `README.md` 或 `docs/progress.md`。
7. 如果阶段目标没完成，不要进入下一阶段。

建议每阶段完成后生成一次 git commit。

commit message 格式：

```text
stage-1: add 2d map and obstacle visualization
stage-2: implement prm planner
stage-3: add k candidate path generation
stage-4: add formation and sheet constraints
stage-5: add formation-aware path selection
stage-6: add trajectory optimization
stage-7: add unicycle simulation
stage-8: add metrics and full demo
```

---

## 9. 测试要求

请尽量写轻量测试，不需要过度复杂。

测试重点：

### `test_collision.py`

- 点在障碍物内应判定为 collision；
- 点在障碍物外应判定为 free；
- 穿过障碍物的线段应判定为 collision；
- 绕开的线段应判定为 free。

### `test_prm.py`

- PRM 能构图；
- 在简单无障碍场景中能找到路径；
- 在完全封闭场景中应返回失败而不是崩溃。

### `test_formation.py`

- 给定 leader pose 能生成 5 个机器人位置；
- sheet 边长约束正常计算；
- 队形误差正常计算。

### `test_trajectory_opt.py`

- 优化后路径端点不变；
- 优化后路径点数量不变；
- 优化失败时有 fallback。

---

## 10. 代码风格要求

1. 使用 type hints。
2. 使用 dataclass 表达简单数据结构。
3. 避免全局变量。
4. 随机数必须支持 seed。
5. 绘图函数与算法函数分离。
6. 不要在算法模块里直接 `plt.show()`，由 main 或 vis 模块控制。
7. 所有路径、输出目录从 config 或参数读取。
8. 对失败情况给出清晰错误信息。

---

## 11. 允许的简化假设

为了让项目能快速落地，允许以下简化：

1. 障碍物只考虑圆形和轴对齐矩形。
2. 机器人第一版可视为点机器人，后续再加半径。
3. sheet 第一版只用几何边长约束，不做真实柔性体动力学。
4. 同伦搜索第一版用 k-shortest paths 近似。
5. 非完整约束第一版用 unicycle 模型近似。
6. 轨迹优化第一版用惩罚函数，不强制做严格约束优化。
7. 不需要复现 ROS/Gazebo 控制接口。

---

## 12. 最终验收标准

最终项目应满足：

1. `python main.py --stage full` 可以运行成功。
2. 生成一张路径规划图。
3. 生成一个机器人运动动画。
4. 输出实验指标 JSON。
5. 至少有一个场景中：
   - 5 个机器人从起点到终点；
   - 无碰撞；
   - 队形基本保持；
   - sheet 约束没有明显违反。
6. README 能让新用户按步骤复现结果。

---

## 13. 重要提醒

不要把目标扩大成完整 CPDOT 重写。当前任务的重点是：

```text
抽取核心逻辑 → Python 简化实现 → 在自定义场景中验证 → 可视化展示 → 输出指标
```

如果遇到复杂问题，请优先保证最小 demo 可运行，再逐步增强。

