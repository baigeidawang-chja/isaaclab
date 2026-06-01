from __future__ import annotations

import pathlib
import time
from collections import defaultdict
from typing import Callable

import numpy as np
import torch


class ConsecStream:
    """Official-like stream: stateless sample + consec + prefix handled by replay."""

    def __init__(
        self,
        replay,
        *,
        batch_size: int,
        base_length: int,
        prefix: int,
        consec: int,
        device: torch.device,
    ):
        self.replay = replay
        self.batch_size = int(batch_size)
        self.base_length = int(base_length)
        self.prefix = int(prefix)
        self.consec = max(1, int(consec))
        self.device = device
        self._state = None
        self._left = 0

    def can_sample(self) -> bool:
        return self.replay.can_sample(self.batch_size, self.base_length + self.prefix)

    def sample(self):
        if self._left <= 0:
            self._left = self.consec
        batch, self._state = self.replay.sample(
            self.batch_size,
            self.base_length + self.prefix,
            self.device,
            stream_state=self._state,
            stride=self.base_length,
        )
        self._left -= 1
        return batch


class Driver:
    """Step-callback driver, close to official driver semantics."""

    def __init__(self, env, agent, carry, obs, *, num_envs: int, device: torch.device):
        self.env = env
        self.agent = agent
        self.carry = carry
        self.obs = obs
        self.num_envs = int(num_envs)
        self.ep_returns = torch.zeros(self.num_envs, device=device)
        self.ep_lengths = torch.zeros(self.num_envs, dtype=torch.int64, device=self.ep_returns.device)

    def run(
        self,
        *,
        steps: int,
        on_step: Callable[[dict, dict, torch.Tensor], None],
        on_episode: Callable[[int, float, int], None],
    ) -> int:
        collected = 0
        while collected < steps:
            with torch.no_grad():
                self.carry, action, policy_info = self.agent.policy(self.carry, self.obs, mode="train")

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated | truncated

            step_data = {k: v for k, v in next_obs.items()}
            if isinstance(action, dict):
                step_data.update(action)
            else:
                step_data["action"] = action
            for k, v in policy_info.items():
                if k.startswith("ctx_") and isinstance(v, torch.Tensor):
                    step_data[k] = v
            step_data["reward"] = reward.unsqueeze(-1) if reward.dim() == 1 else reward
            step_data["is_terminal"] = terminated.unsqueeze(-1) if terminated.dim() == 1 else terminated
            step_data["is_first"] = next_obs["is_first"].unsqueeze(-1) if next_obs["is_first"].dim() == 1 else next_obs["is_first"]
            step_data["is_last"] = next_obs["is_last"].unsqueeze(-1) if next_obs["is_last"].dim() == 1 else next_obs["is_last"]

            on_step(step_data, policy_info, done)

            self.ep_returns += reward
            self.ep_lengths += 1
            if done.any():
                mask = done.bool()
                for i in range(self.num_envs):
                    if mask[i]:
                        on_episode(i, float(self.ep_returns[i].item()), int(self.ep_lengths[i].item()))
                self.ep_returns[mask] = 0
                self.ep_lengths[mask] = 0

            self.obs = next_obs
            collected += self.num_envs
        return collected


class OfficialRunner:
    """Official-like run loop with Driver + Replay + ConsecStream."""

    def __init__(
        self,
        *,
        env,
        eval_env,
        agent,
        buffer,
        logger,
        args,
        dreamer_cfg: dict,
        simulation_app,
        device: torch.device,
        init_obs: dict,
        init_carry,
        encode_stepid: Callable[[torch.Tensor], torch.Tensor],
    ):
        self.env = env
        self.eval_env = eval_env
        self.agent = agent
        self.buffer = buffer
        self.logger = logger
        self.args = args
        self.cfg = dreamer_cfg
        self.simulation_app = simulation_app
        self.device = device
        self.encode_stepid = encode_stepid

        run_cfg = self.cfg.get("run", {})
        self.batch_size = int(self.cfg.get("batch_size", 16))
        self.batch_length = int(self.cfg.get("batch_length", 64))
        self.report_length = int(self.cfg.get("report_length", self.batch_length))
        self.replay_context = int(self.cfg.get("replay_context", 0))
        self.consec_train = int(self.cfg.get("consec_train", 1))
        self.consec_report = int(self.cfg.get("consec_report", 1))
        self.train_ratio = float(run_cfg.get("train_ratio", 32.0))
        self.report_every = int(run_cfg.get("report_every", 0))
        self.report_batches = int(run_cfg.get("report_batches", 1))
        self.eval_every = int(args.eval_every)
        self.eval_eps = int(run_cfg.get("eval_eps", 1))
        self.eval_envs = int(run_cfg.get("eval_envs", 1))

        self.global_step = 0
        self.train_steps = 0
        self.env_step_ids = torch.zeros(args.num_envs, dtype=torch.int64, device=device)
        self.total_episodes = 0
        self.best_eval_score = -float("inf")
        self.next_eval_step = max(self.eval_every, 1) if self.eval_every > 0 else -1
        self.next_report_step = max(self.report_every, 1) if self.report_every > 0 else -1
        self.train_budget = 0.0
        self.start_time = time.time()
        self.metrics_accum = defaultdict(list)

        self.driver = Driver(self.env, self.agent, init_carry, init_obs, num_envs=args.num_envs, device=device)
        self.train_carry = self.agent.init_carry(self.batch_size, self.device)
        self.report_carry = self.agent.init_carry(self.batch_size, self.device)
        self.train_stream = ConsecStream(
            self.buffer,
            batch_size=self.batch_size,
            base_length=self.batch_length,
            prefix=self.replay_context,
            consec=self.consec_train,
            device=self.device,
        )
        self.report_stream = ConsecStream(
            self.buffer,
            batch_size=self.batch_size,
            base_length=self.report_length,
            prefix=self.replay_context,
            consec=self.consec_report,
            device=self.device,
        )

    def _on_episode(self, env_id: int, score: float, length: int):
        self.logger.log_episode(score, length)
        self.total_episodes += 1

    def _on_step(self, step_data: dict, policy_info: dict, done: torch.Tensor):
        step_data["consec"] = self.driver.ep_lengths.unsqueeze(-1).clone()
        step_data["stepid"] = self.encode_stepid(self.env_step_ids)
        self.buffer.add(step_data)
        self.global_step += self.args.num_envs
        self.env_step_ids += 1
        self.logger.step(self.args.num_envs)
        for k, v in policy_info.items():
            if isinstance(v, (int, float)):
                self.metrics_accum[k].append(v)

    def _run_eval_once(self) -> dict:
        if self.eval_env is None:
            return {}
        was_training = self.agent.training
        self.agent.eval()
        ecarry = self.agent.init_carry(self.eval_env.num_envs, self.device)
        eobs, _ = self.eval_env.reset()
        eret = torch.zeros(self.eval_env.num_envs, device=self.device)
        elen = torch.zeros(self.eval_env.num_envs, dtype=torch.int64, device=self.device)
        scores, lengths = [], []
        while len(scores) < self.eval_eps and self.simulation_app.is_running():
            with torch.no_grad():
                ecarry, eaction, _ = self.agent.policy(ecarry, eobs, mode="eval")
            nobs, rew, term, trunc, _ = self.eval_env.step(eaction)
            done = term | trunc
            eret += rew
            elen += 1
            if done.any():
                mask = done.bool()
                for i in range(self.eval_env.num_envs):
                    if mask[i] and len(scores) < self.eval_eps:
                        scores.append(float(eret[i].item()))
                        lengths.append(int(elen[i].item()))
                eret[mask] = 0
                elen[mask] = 0
            eobs = nobs
        if was_training:
            self.agent.train()
        if not scores:
            return {}
        return {
            "eval/score_mean": float(np.mean(scores)),
            "eval/score_max": float(np.max(scores)),
            "eval/length_mean": float(np.mean(lengths)),
            "eval/episodes": float(len(scores)),
        }

    def _train_updates(self):
        self.train_budget += self.args.num_envs * (self.train_ratio / max(self.batch_size * self.batch_length, 1))
        if self.global_step < self.args.prefill_steps or not self.train_stream.can_sample():
            return
        num_updates = int(self.train_budget)
        for _ in range(num_updates):
            batch = self.train_stream.sample()
            if "policy" in batch:
                batch["obs"] = batch["policy"]
            self.train_carry, mets, replay_updates = self.agent.train_step(self.train_carry, batch, debug=False)
            self.train_steps += 1
            self.train_budget -= 1.0
            if "_meta_start" in batch and "_meta_env" in batch and "priority" in mets:
                p = mets["priority"]
                if isinstance(p, torch.Tensor):
                    self.buffer.update_priority(batch["_meta_start"], batch["_meta_env"], p)
            if "_meta_start" in batch and "_meta_env" in batch and replay_updates:
                stepid = replay_updates.get("stepid", None)
                updates = {k: v for k, v in replay_updates.items() if k != "stepid"}
                self.buffer.update_context(batch["_meta_start"], batch["_meta_env"], updates, stepid=stepid)
            for k, v in mets.items():
                if isinstance(v, (int, float)):
                    self.metrics_accum[f"train/{k}"].append(v)

    def _maybe_report(self):
        if self.next_report_step <= 0 or self.global_step < self.next_report_step:
            return
        if not self.report_stream.can_sample():
            return
        racc = defaultdict(list)
        rmedia = {}
        for _ in range(max(1, self.report_batches)):
            batch = self.report_stream.sample()
            if "policy" in batch:
                batch["obs"] = batch["policy"]
            self.report_carry, mets = self.agent.report_step(self.report_carry, batch)
            for k, v in mets.items():
                if isinstance(v, (int, float)):
                    racc[k].append(float(v))
                elif isinstance(v, np.ndarray):
                    rmedia[k] = v
        merged = {k: float(np.mean(v)) for k, v in racc.items() if v}
        merged.update(rmedia)
        if merged:
            self.logger.log_metrics(merged)
        self.next_report_step += self.report_every

    def _maybe_eval(self, logdir: str):
        if self.next_eval_step <= 0 or self.global_step < self.next_eval_step:
            return
        mets = self._run_eval_once()
        if mets:
            self.logger.log_metrics(mets)
            score = float(mets["eval/score_mean"])
            if score > self.best_eval_score:
                self.best_eval_score = score
                ckpt_dir = pathlib.Path(logdir) / "checkpoints"
                ckpt_dir.mkdir(exist_ok=True)
                torch.save(
                    {
                        "agent": self.agent.state_dict(),
                        "optimizer": self.agent.optimizer.state_dict(),
                        "global_step": self.global_step,
                        "train_steps": self.train_steps,
                        "total_episodes": self.total_episodes,
                        "best_eval_score": self.best_eval_score,
                    },
                    ckpt_dir / "agent_best_eval.pt",
                )
        self.next_eval_step += self.eval_every

    def _maybe_log(self):
        if self.global_step % self.args.log_every >= self.args.num_envs:
            return
        elapsed = time.time() - self.start_time
        fps = self.global_step / max(elapsed, 1e-6)
        out = {"fps": fps, "train_steps": self.train_steps, "episodes": self.total_episodes, "buffer_size": self.buffer.total}
        for k, vals in self.metrics_accum.items():
            if vals:
                out[k] = float(np.mean(vals))
        self.metrics_accum.clear()
        if self.logger.episode_scores:
            out.update(
                {
                    "episode/score_mean": float(np.mean(self.logger.episode_scores[-100:])),
                    "episode/score_max": float(np.max(self.logger.episode_scores[-100:])),
                    "episode/length_mean": float(np.mean(self.logger.episode_lengths[-100:])),
                }
            )
        self.logger.log_metrics(out)

    def _maybe_save(self, logdir: str):
        if self.global_step % self.args.save_every >= self.args.num_envs:
            return
        ckpt_dir = pathlib.Path(logdir) / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        torch.save(
            {
                "agent": self.agent.state_dict(),
                "optimizer": self.agent.optimizer.state_dict(),
                "global_step": self.global_step,
                "train_steps": self.train_steps,
                "total_episodes": self.total_episodes,
            },
            ckpt_dir / f"agent_{self.global_step}.pt",
        )

    def run(self, *, max_steps: int, logdir: str):
        collect_steps = self.args.num_envs  # one vector-env step per outer loop
        while self.global_step < max_steps and self.simulation_app.is_running():
            self.driver.run(steps=collect_steps, on_step=self._on_step, on_episode=self._on_episode)
            self._train_updates()
            self._maybe_log()
            self._maybe_save(logdir)
            self._maybe_eval(logdir)
            self._maybe_report()
