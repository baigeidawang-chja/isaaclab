from __future__ import annotations

from collections.abc import Sequence

import torch


def _resolve_env_ids(env, env_ids: Sequence[int] | torch.Tensor | slice | None) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)
    if not isinstance(env_ids, torch.Tensor):
        return torch.tensor(env_ids, device=env.device, dtype=torch.long)
    return env_ids.to(device=env.device, dtype=torch.long)


def reset_runtime_buffers(env, env_ids: Sequence[int] | torch.Tensor | slice | None):
    """Clear per-episode runtime buffers owned by this task."""
    env_ids = _resolve_env_ids(env, env_ids)

    if hasattr(env, "action_manager"):
        action = env.action_manager.action
        if not hasattr(env, "_tpc_prev_action") or env._tpc_prev_action.shape != action.shape:
            env._tpc_prev_action = torch.zeros_like(action)
        else:
            env._tpc_prev_action[env_ids] = 0.0

        for term_name in env.action_manager.active_terms:
            term = env.action_manager.get_term(term_name)
            if hasattr(term, "reset"):
                term.reset(env_ids=env_ids)
