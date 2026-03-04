# RL-Control-Jet：case3 / case4 使用说明

## 1. 项目目标
本项目通过 PPO 强化学习控制 Fluent 边界入口速度，目标是降低总压损失（`tploss`）。

- `case3 no sin.py`：无正弦调制，仅控制速度幅值 `A`
- `case 4 no sin.py`：控制幅值 `A` + 频率 `f`，速度表达式含 `sin(...)`

---

## 2. 依赖环境
- Python 3.8+
- `gym`（当前代码是 Gym 0.21 风格接口）
- `stable-baselines3`
- `numpy`, `pandas`
- `ansys.fluent.core` (PyFluent)
- 本地可用的 ANSYS Fluent 求解环境和许可证

---

## 3. 输入文件准备
每个 case 至少需要：
1. `.cas.h5`（网格/设置）
2. `.dat.h5`（初场）
3. Fluent 运行后可输出的 report 文件（代码默认匹配 `report-def-0-rfile*`）

---

## 4. 核心流程（两个脚本通用）
1. 创建 `CompressorEnv`
2. 在 `__init__` 启动 Fluent：
   - `launch_fluent(mode="solver", dimension=3, ...)`
   - `read_case(...)`
   - 设置 `reporting_interval` 和 `profile_update_interval`
3. `reset()`：
   - 读取 `.dat` 初场
   - 设置初始入口速度表达式
   - 返回初始 observation
4. `step(action)`：
   - 动作归一化值映射到物理量（A 或 A+f）
   - 写入入口速度边界
   - `dual_time_iterate(time_step_count=slice_len, max_iter_per_step=30)`
   - 读取最新 `tploss`
   - 计算 reward：`baseline_tploss*10 - tploss_now`
   - 返回 `(obs, reward, done, info)`
5. 使用 SB3 `PPO.learn(total_timesteps=...)` 训练
6. `model.save(...)` 保存模型

---

## 5. case3 说明（单动作）
- 动作空间：`Box(shape=(1,), [0,1])`
- 物理映射：`A = A_min + a_norm*(A_max-A_min)`
- 入口表达式：`"{A}[m s^-1]"`
- 观测：`[a_norm, tploss_now, t_idx/max_decisions]`

---

## 6. case4 说明（双动作）
- 动作空间：`Box(shape=(2,), [0,1]^2)`
- 物理映射：
  - `A = A_min + a_norm*(A_max-A_min)`
  - `f = f_min + f_norm*(f_max-f_min)`
- 入口表达式：`"{A}[m s^-1]*sin(2*pi*f*(t-t0))"`
- 观测：`[a_norm, f_norm, tploss_now, t_idx/max_decisions]`

---

## 7. 关键超参数（当前代码）
- 算法：PPO (`MlpPolicy`)
- 网络：`[64, 128, 64]`
- `learning_rate=5e-4`
- `n_epochs=5`
- `clip_range=0.2`
- `gae_lambda=0.95`
- `ent_coef=0.01`
- case3: `n_steps=80`, case4: `n_steps=40`

---

## 8. 训练与续训
当前脚本会先构建一个 PPO，再尝试 `PPO.load(...)` 覆盖。
建议使用“模型文件存在则加载，否则新建”方式以支持首跑与断点续训。

---

## 9. 输出与调试
- 每步 `info` 包含：`tploss_now`, `A`, `f(若有)`, `expr`, `t_idx`, `reward_now`
- 终止条件：`decision_count >= max_decisions`
- 训练后模型保存为 `my_model_nosin5`（脚本中可改）

---

## 10. 常见问题
1. 找不到 case/data 文件：检查路径与工作目录
2. 无法读取 `report-def-0-rfile*`：确认 Fluent report 定义和输出路径
3. 模型加载失败：检查 `PPO.load()` 的模型名是否存在
4. 训练不稳定：优先检查 reset 初始观测是否与真实场一致、reward 尺度是否合理
