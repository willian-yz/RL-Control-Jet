# dt = 3e-5s
# 1000dt 5个周期 40个动作 25dt做一次动作
#

from pathlib import Path
import os
import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
import ansys.fluent.core as pyfluent
# from sb3_contrib import RecurrentPPO
import random
from pathlib import Path
import re
import pandas as pd

os.chdir(Path(r"D:\LYZ\A case3 with hole jet ag70"))

class CompressorEnv(gym.Env):
    """
    连续控制：每 decision_interval 个物理时间步执行一次动作（action hold），
    目标是降低出口总压损失 tploss。基于 Ansys Fluent 的耦合 CFD 环境。

    观测（示例，4维）:
      [A_norm, f_norm, tploss_now_norm, d_tploss_norm]
      - A_norm:    当前幅值（归一化到 [0,1]）
      - f_norm:    当前频率（归一化到 [0,1]）
      - tploss_norm:            当前总压损失（可直接用原值，建议配合 VecNormalize）
      - d_tploss_norm:          决策周期内均值(或末值) 与上一周期的差分

    动作（2维）:
      [A_norm, f_norm] ∈ [0,1]^2
      将线性映射到物理空间：
        A = A_min + A_norm * (A_max - A_min)
        f = f_min + f_norm * (f_max - f_min)

    终止：
      - 固定决策步数 max_decisions 后结束（推荐确定性）
      - 或者基于 tploss 的异常阈值（可选）

    注：本环境保持 Gym 0.21 的接口：step 返回 (obs, reward, done, info)
    """

    def __init__(
            self,
            cas_path: str,  # cas_path
            data_path: str = "70 angle unsteady_jet0_dt=0.03s.dat.h5",  # data_path
            show_gui: bool = True,
            processor_count: int = 56,  # 运行核心
            slice_len: int = 10,  # 每次采样推进的物理步
            A_range=(0.0, 200.0),  # m/s
            baseline_tploss: float = 0.07,  # 若提供，将在奖励中做锚定
            reward_scales=(0.2, 0.75, 0.1),  # (k_improve, k_anchor, k_smooth)
            report_def="totalw",
            inlet_name="hole_inlet",
            pi=3.1415926,  # 修正 π
    ):
        super().__init__()
        self.cas_path = cas_path
        self.data_path = data_path
        self.show_gui = show_gui
        self.processor_count = processor_count
        self.slice_len = slice_len
        self.A_min, self.A_max = A_range
        self.t_idx = 0
        self.baseline_tploss = baseline_tploss
        self.inlet_name = inlet_name
        self.pi = pi

        # --- Fluent 会话 修改参数 ---
        self.session = pyfluent.launch_fluent(
            mode="solver", dimension=3, show_gui=self.show_gui, processor_count=self.processor_count
        )
        self.session.settings.file.read_case(file_name=self.cas_path)
        # 统计/更新频率（内部时间步）
        self.session.solution.run_calculation.reporting_interval = self.slice_len
        self.session.solution.run_calculation.profile_update_interval = self.slice_len

        # --- Gym 空间定义 ---
        # 动作：A_norm, f_norm ∈ [0,1]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # 观测：A_norm, tploss_av25 * 10, t_idx
        # 注意：tploss 值域可用 VecNormalize 做在线归一化，这里先给个宽松边界
        obs_low = np.array([0.0, -np.inf, 0], dtype=np.float32)
        obs_high = np.array([1.0, np.inf, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)  # 连续动作

        # --- 内部状态 ---
        self.A = 0.0
        self.t0 = 0.210084  # 初始参考时间（来自你的代码）
        self.current_velocity_expr = "0[m s^-1]"
        self.last_tploss = None
        self.last_action = np.zeros(1, dtype=np.float32)

        self.max_decisions = 80  # 决策40次
        self.decision_count = 0

        # 历史（仅本实例，用于调试；避免全局共享导致并行冲突）
        self._amplitude_hist = []
        self._frequency_hist = []
        self._tploss_hist = []
        self._velocity_expr_hist = []

    # ---- 工具函数 ----
    def _norm_to_phys(self, a_norm):
        A = float(self.A_min + np.clip(a_norm, 0.0, 1.0) * (self.A_max - self.A_min))
        return A

    def _velocity_expr(self, A):
        A = f"{float(A):.2f}"
        return f"{A}[m s^-1]"

    def _set_velocity(self, expr):
        self.session.setup.boundary_conditions.velocity_inlet[self.inlet_name].momentum.velocity.value = expr

    def _iterate(self, time_step_count):
        self.session.settings.solution.run_calculation.dual_time_iterate(
            time_step_count=time_step_count, max_iter_per_step=30
        )

    def _get_tploss(self):
        folder = Path()  # 当前目录

        base = "report-def-0-rfile"
        patterns = [base, base + "_*", base + "_*_*"]

        # 收集候选文件（去重）
        files = set()
        for pat in patterns:
            files.update(folder.glob(pat))
        files = [p for p in files if p.is_file()]

        if not files:
            raise FileNotFoundError(f"No files match: {patterns} in {folder.resolve()}")

        # 选“最新”：按文件修改时间
        latest_file = max(files, key=lambda p: p.stat().st_mtime)

        # 读取最后一行第二列（你原来的逻辑）
        read_total_w = np.array(
            pd.read_csv(latest_file, sep=r"\s+", skiprows=2).iloc[-1, 2]
        )

        print("Latest:", latest_file)
        print(read_total_w)

        return read_total_w

    # ---- Gym 接口 ----
    def seed(self, seed=2025):
        return seed

    def reset(self):
        # 读入初场数据
        self.session.settings.file.read_data(file_name=self.data_path)

        self.A = self._norm_to_phys(0.0)
        self.current_velocity_expr = self._velocity_expr(self.A)
        self._set_velocity(self.current_velocity_expr)
        tploss_now = float(0.07) * 10

        # 观测：A_norm, f_norm, tploss, t_idx
        obs = np.array([0.0, tploss_now, 0], dtype=np.float32)

        # 历史记录
        self._amplitude_hist = [self.A]
        self._tploss_hist = [tploss_now]
        self._velocity_expr_hist = [self.current_velocity_expr]
        self.t_idx = 0
        self.decision_count = 0
        return obs

    def step(self, action):
        """
        动作在整个决策周期内保持（action hold），周期内每 slice_len 步采样一次 tploss，
        最终用“均值 tploss” 与 “上一周期均值”构造奖励。
        """
        done = False
        self.decision_count += 1
        # 动作归一化 -> 物理量
        a_norm = float(np.clip(action[0], 0.0, 1.0))
        A = self._norm_to_phys(a_norm)

        # 设置边界条件（保持 decision_interval 个时间步）
        expr = self._velocity_expr(A)
        self.current_velocity_expr = expr
        self._set_velocity(expr)

        # 运行 25 dt
        self._iterate(self.slice_len)
        # 直接获得25 dt后的tploss
        tploss_now = float(self._get_tploss()) * 10

        # --- 奖励构造 ---
        # 25个时间步长的reward小于平均损失
        reward = (self.baseline_tploss * 10 - tploss_now)

        # 更新内部状态
        self.A = A
        self.last_tploss = tploss_now
        self.last_action = np.array([a_norm], dtype=np.float32)
        self.t_idx += 1
        # 观测
        obs = np.array([a_norm, tploss_now, self.t_idx / self.max_decisions], dtype=np.float32)

        # 历史
        self._amplitude_hist.append(A)
        self._velocity_expr_hist.append(expr)

        # 终止条件：决策40次，5个周期
        if self.decision_count >= self.max_decisions:
            done = True

        info = {
            "tploss_now": tploss_now,
            "A": A,
            "expr": expr,
            "t_idx": self.t_idx,
            "reward_now": reward
        }
        # （如需把周期内采样的明细暴露给外部，也可加到 info）
        print(info, '\n')
        return obs, float(reward), done, info

    def close(self):
        if hasattr(self, "session") and self.session is not None:
            try:
                self.session.exit()
            except Exception:
                pass


env = DummyVecEnv([lambda: CompressorEnv('70 angle unsteady_jet0_dt=0.03s.cas.h5')])
# 初始化PPO算法
log_dir = './ppo_logs_new'
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    policy_kwargs=dict(
        net_arch = [64,128,64],
    ),
    learning_rate=5e-4,
    n_steps=80,
    batch_size=20,
    n_epochs=5,
    clip_range=0.2,
    gae_lambda=0.95,
    ent_coef=0.01
)

model = PPO.load("my_model_nosin4",env=env)

#obs = env.reset()
#episode_reward = 0
#for step in range (80):
#     action, _states = model.predict(obs)
#     obs, reward, done, info = env.step(action)
#     episode_reward += float(reward)
#     print(step,reward,info,"\n")

model.learn(total_timesteps=100*10)
model.save('my_model_nosin5')
#model.learn(total_timesteps=100*10)
#model.save('my_model_nosin6')