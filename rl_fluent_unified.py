"""Unified RL-Fluent workflow for 1D/2D action control.

- 2D action: control amplitude A and frequency f
  velocity expression: A*sin(2*pi*f*(t-t0))
- 1D action: control only amplitude A
  velocity expression: A

This script keeps Gym 0.21 style step/reset to match stable-baselines3 usage in
existing project scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse

import gym
from gym import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import ansys.fluent.core as pyfluent


@dataclass
class EnvConfig:
    cas_path: str
    data_path: str
    action_dim: int = 2  # 1 or 2
    show_gui: bool = True
    processor_count: int = 56
    slice_len: int = 10
    max_decisions: int = 100
    inlet_name: str = "hole_inlet"
    baseline_tploss: float = 0.12
    t0: float = 0.210084
    pi: float = 3.1415926
    a_range: tuple[float, float] = (0.0, 100.0)
    f_range: tuple[float, float] = (200.0, 2000.0)
    report_pattern: str = "report-def-0-rfile"
    report_dir: str = "."


@dataclass
class TrainConfig:
    model_path: str = "my_model_unified"
    save_path: str = "my_model_unified_next"
    total_timesteps: int = 1000
    learning_rate: float = 5e-4
    n_steps: int = 80
    batch_size: int = 20
    n_epochs: int = 5
    clip_range: float = 0.2
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    net_arch: tuple[int, ...] = (64, 128, 64)


class CompressorEnv(gym.Env):
    """CFD coupled RL environment supporting both 1D and 2D actions."""

    def __init__(self, cfg: EnvConfig):
        super().__init__()
        self.cfg = cfg
        if self.cfg.action_dim not in (1, 2):
            raise ValueError(f"action_dim must be 1 or 2, got {self.cfg.action_dim}")

        self.a_min, self.a_max = self.cfg.a_range
        self.f_min, self.f_max = self.cfg.f_range
        self.decision_count = 0
        self.t_idx = 0
        self.last_tploss = None
        self.last_action = np.zeros(self.cfg.action_dim, dtype=np.float32)

        self.session = self._launch_fluent()
        self._configure_run_intervals()

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.cfg.action_dim,),
            dtype=np.float32,
        )

        if self.cfg.action_dim == 1:
            # [a_norm, tploss, progress]
            obs_low = np.array([0.0, -np.inf, 0.0], dtype=np.float32)
            obs_high = np.array([1.0, np.inf, 1.0], dtype=np.float32)
        else:
            # [a_norm, f_norm, tploss, progress]
            obs_low = np.array([0.0, 0.0, -np.inf, 0.0], dtype=np.float32)
            obs_high = np.array([1.0, 1.0, np.inf, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    def _launch_fluent(self):
        session = pyfluent.launch_fluent(
            mode="solver",
            dimension=3,
            show_gui=self.cfg.show_gui,
            processor_count=self.cfg.processor_count,
        )
        session.settings.file.read_case(file_name=self.cfg.cas_path)
        return session

    def _configure_run_intervals(self):
        self.session.solution.run_calculation.reporting_interval = self.cfg.slice_len
        self.session.solution.run_calculation.profile_update_interval = self.cfg.slice_len

    def _norm_to_amplitude(self, a_norm: float) -> float:
        return float(self.a_min + np.clip(a_norm, 0.0, 1.0) * (self.a_max - self.a_min))

    def _norm_to_frequency(self, f_norm: float) -> float:
        return float(self.f_min + np.clip(f_norm, 0.0, 1.0) * (self.f_max - self.f_min))

    def _build_velocity_expr(self, a_norm: float, f_norm: float | None = None) -> str:
        amplitude = self._norm_to_amplitude(a_norm)
        if self.cfg.action_dim == 1:
            return f"{amplitude:.2f}[m s^-1]"

        if f_norm is None:
            raise ValueError("f_norm is required when action_dim=2")
        frequency = self._norm_to_frequency(f_norm)
        return (
            f"{amplitude:.2f}[m s^-1]*sin(2*{self.cfg.pi}*{frequency:.2f} [s^-1]"
            f"*(t - {self.cfg.t0}[s]))"
        )

    def _set_velocity(self, expr: str):
        self.session.setup.boundary_conditions.velocity_inlet[
            self.cfg.inlet_name
        ].momentum.velocity.value = expr

    def _iterate(self):
        self.session.settings.solution.run_calculation.dual_time_iterate(
            time_step_count=self.cfg.slice_len,
            max_iter_per_step=30,
        )

    def _get_latest_report_value(self) -> float:
        folder = Path(self.cfg.report_dir)
        base = self.cfg.report_pattern
        patterns = [base, f"{base}_*", f"{base}_*_*"]

        files = set()
        for pat in patterns:
            files.update(folder.glob(pat))
        files = [p for p in files if p.is_file()]
        if not files:
            raise FileNotFoundError(f"No files match {patterns} in {folder.resolve()}")

        latest = max(files, key=lambda p: p.stat().st_mtime)
        value = float(pd.read_csv(latest, sep=r"\s+", skiprows=2).iloc[-1, 2])
        return value

    def seed(self, seed=2025):
        return seed

    def reset(self):
        self.session.settings.file.read_data(file_name=self.cfg.data_path)

        if self.cfg.action_dim == 1:
            expr = self._build_velocity_expr(0.0)
            self._set_velocity(expr)
            tploss_now = self.cfg.baseline_tploss * 10
            obs = np.array([0.0, tploss_now, 0.0], dtype=np.float32)
        else:
            expr = self._build_velocity_expr(0.0, 0.0)
            self._set_velocity(expr)
            tploss_now = self.cfg.baseline_tploss * 10
            obs = np.array([0.0, 0.0, tploss_now, 0.0], dtype=np.float32)

        self.decision_count = 0
        self.t_idx = 0
        self.last_tploss = tploss_now
        self.last_action = np.zeros(self.cfg.action_dim, dtype=np.float32)
        return obs

    def step(self, action):
        self.decision_count += 1

        if self.cfg.action_dim == 1:
            a_norm = float(np.clip(action[0], 0.0, 1.0))
            expr = self._build_velocity_expr(a_norm)
            self._set_velocity(expr)
            self._iterate()
            tploss_now = self._get_latest_report_value() * 10
            reward = self.cfg.baseline_tploss * 10 - tploss_now
            self.last_action = np.array([a_norm], dtype=np.float32)
            self.t_idx += 1
            obs = np.array([a_norm, tploss_now, self.t_idx / self.cfg.max_decisions], dtype=np.float32)
            info = {"A_norm": a_norm, "expr": expr, "tploss_now": tploss_now, "reward_now": reward}
        else:
            a_norm = float(np.clip(action[0], 0.0, 1.0))
            f_norm = float(np.clip(action[1], 0.0, 1.0))
            expr = self._build_velocity_expr(a_norm, f_norm)
            self._set_velocity(expr)
            self._iterate()
            tploss_now = self._get_latest_report_value() * 10
            reward = self.cfg.baseline_tploss * 10 - tploss_now
            self.last_action = np.array([a_norm, f_norm], dtype=np.float32)
            self.t_idx += 1
            obs = np.array(
                [a_norm, f_norm, tploss_now, self.t_idx / self.cfg.max_decisions],
                dtype=np.float32,
            )
            info = {
                "A_norm": a_norm,
                "f_norm": f_norm,
                "expr": expr,
                "tploss_now": tploss_now,
                "reward_now": reward,
            }

        done = self.decision_count >= self.cfg.max_decisions
        return obs, float(reward), done, info

    def close(self):
        if getattr(self, "session", None) is not None:
            self.session.exit()


def build_model(env: DummyVecEnv, train_cfg: TrainConfig) -> PPO:
    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        policy_kwargs={"net_arch": list(train_cfg.net_arch)},
        learning_rate=train_cfg.learning_rate,
        n_steps=train_cfg.n_steps,
        batch_size=train_cfg.batch_size,
        n_epochs=train_cfg.n_epochs,
        clip_range=train_cfg.clip_range,
        gae_lambda=train_cfg.gae_lambda,
        ent_coef=train_cfg.ent_coef,
    )


def load_or_create_model(env: DummyVecEnv, train_cfg: TrainConfig) -> PPO:
    model_path = Path(train_cfg.model_path)
    if model_path.exists() or model_path.with_suffix(".zip").exists():
        print(f"[INFO] Resume training from model: {train_cfg.model_path}")
        return PPO.load(train_cfg.model_path, env=env)

    print("[INFO] No existing model found. Start training from scratch.")
    return build_model(env, train_cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Fluent RL trainer")
    parser.add_argument("--cas-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--action-dim", type=int, default=2, choices=[1, 2])
    parser.add_argument("--t0", type=float, default=0.210084)
    parser.add_argument("--baseline-tploss", type=float, default=0.12)
    parser.add_argument("--slice-len", type=int, default=10)
    parser.add_argument("--max-decisions", type=int, default=100)
    parser.add_argument("--model-path", default="my_model_unified")
    parser.add_argument("--save-path", default="my_model_unified_next")
    parser.add_argument("--total-timesteps", type=int, default=1000)
    parser.add_argument("--show-gui", action="store_true")
    parser.add_argument("--processor-count", type=int, default=56)
    parser.add_argument("--report-dir", default=".")
    parser.add_argument("--report-pattern", default="report-def-0-rfile")
    return parser.parse_args()


def main():
    args = parse_args()

    env_cfg = EnvConfig(
        cas_path=args.cas_path,
        data_path=args.data_path,
        action_dim=args.action_dim,
        t0=args.t0,
        baseline_tploss=args.baseline_tploss,
        slice_len=args.slice_len,
        max_decisions=args.max_decisions,
        show_gui=args.show_gui,
        processor_count=args.processor_count,
        report_dir=args.report_dir,
        report_pattern=args.report_pattern,
    )
    train_cfg = TrainConfig(
        model_path=args.model_path,
        save_path=args.save_path,
        total_timesteps=args.total_timesteps,
        n_steps=args.max_decisions,
    )

    env = DummyVecEnv([lambda: CompressorEnv(env_cfg)])
    model = load_or_create_model(env, train_cfg)
    model.learn(total_timesteps=train_cfg.total_timesteps)
    model.save(train_cfg.save_path)
    print(f"[INFO] Model saved to: {train_cfg.save_path}")


if __name__ == "__main__":
    main()
