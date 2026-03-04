# Unified Fluent RL Workflow 使用说明

这个文档对应 `rl_fluent_unified.py`，用于把你现在的 case3/case4 工作流合并到一个更整洁的脚本中。

## 功能概览

- **支持 2 维动作**（`action_dim=2`）
  - 控制 `A`（幅值）和 `f`（频率）
  - 速度表达式：`A*sin(2*pi*f*(t - t0))`
- **支持 1 维动作**（`action_dim=1`）
  - 只控制 `A`
  - 速度表达式：`A`
- **支持模型存在性判断**
  - 如果 `--model-path` 已存在，则自动加载继续训练
  - 否则从头创建新模型训练
- **main 入口参数化**
  - 关键参数（动作维度、`t0`、baseline loss、训练步数等）都可从命令行设置

---

## 脚本结构说明

`rl_fluent_unified.py` 主要由这几部分组成：

1. `EnvConfig`（环境参数）
   - 例如 `action_dim`、`t0`、`baseline_tploss`、`slice_len`、`max_decisions`
2. `TrainConfig`（训练参数）
   - 例如 `model_path`、`save_path`、`total_timesteps`
3. `CompressorEnv`（Gym 环境）
   - 在 `__init__` 里启动 Fluent + 读取 case
   - `reset()` 读取 data 并设置初始边界
   - `step(action)` 执行动作、推进 Fluent、读取 `tploss`、计算 reward
4. `load_or_create_model()`
   - 统一处理“加载已有模型 / 新建模型”
5. `main()`
   - 只负责读参数、构造配置、运行训练

---

## 奖励与观测

- Reward 当前实现：
  - `reward = baseline_tploss*10 - tploss_now`
- 观测：
  - 1D 动作：`[a_norm, tploss_now, progress]`
  - 2D 动作：`[a_norm, f_norm, tploss_now, progress]`

其中 `progress = t_idx / max_decisions`。

---

## 运行前准备

请确保：

1. Python 环境已安装依赖：
   - `gym`, `stable-baselines3`, `numpy`, `pandas`, `ansys.fluent.core`
2. Fluent 许可证可用
3. 你有可用的 `.cas.h5` 和 `.dat.h5` 文件
4. Fluent report 文件输出规则与脚本参数一致（默认 `report-def-0-rfile*`）

---

## 使用方式

## 1) 2维动作训练（A + f）

```bash
python rl_fluent_unified.py \
  --cas-path "double_sided 70 angle steady_jet0_dt=0.0115s.cas.h5" \
  --data-path "double_sided 70 angle steady_jet0_dt=0.0115s.dat.h5" \
  --action-dim 2 \
  --t0 0.210084 \
  --baseline-tploss 0.12 \
  --max-decisions 100 \
  --total-timesteps 1000 \
  --model-path my_model_unified \
  --save-path my_model_unified_next
```

## 2) 1维动作训练（仅 A）

```bash
python rl_fluent_unified.py \
  --cas-path "70 angle unsteady_jet0_dt=0.03s.cas.h5" \
  --data-path "70 angle unsteady_jet0_dt=0.03s.dat.h5" \
  --action-dim 1 \
  --t0 0.210084 \
  --baseline-tploss 0.07 \
  --max-decisions 80 \
  --total-timesteps 1000 \
  --model-path my_model_unified \
  --save-path my_model_unified_next
```

---

## 模型存在性判断逻辑

脚本会检查：

- `model_path`
- `model_path.zip`

只要任一存在，就执行：

- `PPO.load(model_path, env=env)`

否则执行：

- 新建 PPO 模型并从头训练

---

## 可调关键参数（建议优先调这些）

- `--action-dim`：动作维度（1 或 2）
- `--t0`：速度表达式中的时间偏移
- `--baseline-tploss`：奖励锚点
- `--slice-len`：每个决策推进的时间步数
- `--max-decisions`：每个 episode 决策次数上限
- `--total-timesteps`：总训练步数
- `--processor-count`：Fluent 并行核数

---

## 注意事项

1. 当前 `reset()` 中初始 `tploss` 使用的是 `baseline_tploss*10`，如果你希望更物理一致，可在 reset 后推进一步并读取真实 report 值。
2. 当前 `report` 读取策略是读取匹配模式下最新文件，建议保证工作目录干净，或者后续改成单次训练固定 report 文件路径。
3. 这是 Gym 0.21 风格接口（`step -> obs, reward, done, info`），和你现有脚本保持一致。
