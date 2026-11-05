# Known issues

Keep concrete, reproduced issues here.

Recommended format:

## <short issue title>
- Reproduction command:
- Observed behavior:
- Expected behavior:
- Relevant files:
- Last known good commit/checkpoint:
- Current hypothesis:
- Status:

Do not add speculative issues that have not been reproduced.

## Deterministic PPO steering target chatters after 0.243-rad scale alignment

- Reproduction command: evaluate
  `2026-10-29_11-38-16_ppo_torch_medium_500_from_short_steer_scale_0243/checkpoints/best_agent.pt`
  with the four `scripts/user/eval_*.py` CommandFollow evaluators.
- Observed behavior: raw steering reaches `|action| >= 1` on 38% to 46% of
  evaluation steps and changes sign on 52% to 57% of steps. The rate limiter
  reduces, but does not remove, the resulting executed steering activity.
- Expected behavior: deterministic mean actions should track heading without
  sustained boundary-to-boundary switching.
- Relevant files: `command_follow_env_cfg.py`, `mdp/actions.py`,
  `skrl_ppo_cfg.yaml`, and `scripts/user/command_follow_evaluation.py`.
- Last known good commit/checkpoint: none established for the revised
  `0.243 rad` action scale.
- Current hypothesis: no directional execution bias is present; policy-output
  sensitivity, observation scaling, and insufficient convergence remain to be
  separated experimentally.
- Status: open; reproduced on all four evaluations, with no NaN or unexpected
  termination.
