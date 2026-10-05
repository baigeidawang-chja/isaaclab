# AGENTS.md

## Project overview

This repository is an Ubuntu-based wheeled-robot reinforcement-learning research project built on NVIDIA Isaac Lab.

The project studies command following, low-traction locomotion, proprioceptive state estimation, failure prediction, and protective control for a wheeled robot.

The reinforcement-learning algorithm is **not fixed to DreamerV3**.

Different research stages may use different algorithms or training stacks, including:

- skrl PPO
- recurrent PPO variants
- DreamerV3
- other reinforcement-learning or model-based methods when explicitly required

Do not assume that DreamerV3 is the default algorithm for the repository.

---

## Current development phase

The current active development phase is:

**Phase 1 — CommandFollow baseline**

Goal:

Train the wheeled robot to follow planner-provided speed and heading commands on a flat, high-traction surface.

This phase exists to validate the basic closed loop:

planner command  
→ policy observation  
→ policy action  
→ rate-limited vehicle execution  
→ vehicle response  
→ command-tracking reward

Do not introduce later-stage traction estimation, failure prediction, or protective-control mechanisms into Phase 1 unless explicitly requested.

---

## Current research namespace

New development for this research line belongs under:

`isaaclab_tasks.user.traction_protective_control`

Expected structure:

```text
source/isaaclab_tasks/isaaclab_tasks/user/traction_protective_control/
  __init__.py
  mdp/
    __init__.py
    actions.py
    commands.py
    observations.py
    rewards.py
    terminations.py
    events.py
  config/
    __init__.py
    car4wd/
      __init__.py
      command_follow_env_cfg.py
      agents/
        __init__.py
        skrl_ppo_cfg.yaml
```

Do not continue adding new research features under:

- `isaaclab_tasks.user.dreamerv3`
- the legacy `user/project` FailureAware namespace

Existing code in those locations may be inspected and reused as references when appropriate, but new Phase 1 implementation should remain logically independent.

Do not move or delete legacy DreamerV3 or `user/project` code unless explicitly requested.

---

## Current Phase 1 task

Primary training task:

`Isaac-TractionProtect-CommandFollow-Car4WD-v0`

Primary play task:

`Isaac-TractionProtect-CommandFollow-Car4WD-Play-v0`

These task IDs describe the current development phase only.

Do not treat them as permanent repository-wide task IDs.

---

## Current Phase 1 training stack

Phase 1 uses:

- `ManagerBasedRLEnv`
- skrl
- MLP-PPO

Training entry point:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py
```

Play / evaluation entry point:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py
```

Typical training command:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task Isaac-TractionProtect-CommandFollow-Car4WD-v0 \
  --num_envs 1024 \
  --algorithm PPO
```

Typical play command:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
  --task Isaac-TractionProtect-CommandFollow-Car4WD-Play-v0 \
  --checkpoint <checkpoint>
```

The current PPO configuration belongs in:

```text
traction_protective_control/config/car4wd/agents/skrl_ppo_cfg.yaml
```

Do not interpret this PPO entry point or configuration as a repository-wide algorithm restriction.

If a later task explicitly uses DreamerV3, recurrent PPO, or another algorithm, inspect that task's own training entry point and configuration instead of forcing the skrl PPO workflow onto it.

---

## Phase 1 architecture

### Robot asset

Reuse:

```python
isaaclab_assets.CAR_CFG
```

The robot configuration is currently based on the existing Car4WD asset.

Prefer reusing the existing robot model and actuator configuration rather than duplicating the asset definition.

Do not modify the upstream robot asset solely to implement Phase 1 command-following behavior unless necessary.

---

## Commands

Phase 1 uses planner-style commands:

```text
[desired_speed, desired_heading]
```

The policy-facing planner command should use:

```text
[desired_speed, heading_error]
```

where:

```text
heading_error = wrap_to_pi(desired_heading - current_heading)
```

Initial recommended command ranges:

```text
desired_speed: 0.3 to 1.2
desired_heading: -0.4 to 0.4 rad
```

Command resampling should initially remain slow enough for the vehicle to establish a meaningful tracking response.

Recommended initial interval:

```text
6 to 10 seconds
```

Play configuration may use fixed or narrower commands to make behavior easier to inspect.

---

## Observations

Phase 1 policy observations should contain only information that is physically available or intentionally exposed to the controller.

Expected observation groups include:

- planner command:
  - desired speed
  - heading error
- base linear velocity in the body frame
- base angular velocity in the body frame
- projected gravity
- IMU state
- four wheel velocities
- previous action or rate-limited executed action

Wheel observations should explicitly select the four wheel joints defined by the current `CAR_CFG`.

Phase 1 observations must not expose privileged future-stage information such as:

- friction coefficient
- terrain type
- slip label
- stuck label
- failure label
- abort-required label
- front/rear traction label
- obstacle information
- waypoint information
- hidden simulator-only failure state

If privileged information is required for debugging, keep it outside the policy observation unless explicitly requested.

---

## Actions

The Phase 1 action semantics are:

```text
[executed_speed, steering_control]
```

Use a rate-limited continuous vehicle action implementation based on the existing `RateLimitedCarVWActionCfg` design where appropriate.

The implementation should remain independent of DreamerV3-specific action classes.

Expected wheel joints:

```text
joint_front_right_wheel_link_wheel
joint_front_left_wheel_link_wheel
joint_back_right_wheel_link_wheel
joint_back_left_wheel_link_wheel
```

Expected steering joints:

```text
joint_front_right_steer
joint_front_left_steer
```

Initial action configuration:

```text
scale = (1.5, 0.8)
offset = (0.0, 0.0)
bounding_strategy = "clip"
no_reverse = False
```

Do not remove reverse capability.

The action space should preserve future ability to:

- actively slow down
- stop
- reverse
- perform protective intervention

Prefer rate limiting after action bounding so the actual executed action remains physically meaningful and inspectable.

---

## Rewards

Phase 1 rewards should evaluate command following and basic motion quality.

Primary reward terms:

### Speed tracking

Reward the projection of world-frame vehicle velocity onto the desired command heading for matching the desired speed.

Do not simply reward raw forward body velocity if doing so conflicts with the commanded heading.

### Heading tracking

Reward alignment between vehicle heading and desired heading.

Use wrapped angular error.

### Upright stability

Penalize excessive roll/pitch or equivalent projected-gravity deviation.

### Action smoothness

Penalize excessive change in control action.

Prefer evaluating the actually executed rate-limited action when the purpose is physical control smoothness.

Phase 1 must not include rewards for:

- waypoint navigation progress
- obstacle avoidance
- collision recovery
- stuck recovery
- low-traction recovery
- backward escape behavior
- terrain traversal
- slip prediction
- failure prediction
- protective intervention

unless explicitly requested.

---

## Terminations

Phase 1 should keep termination logic minimal.

Expected termination conditions:

- time out
- vehicle flip
- optional wide out-of-bounds protection

Out-of-bounds termination exists only to prevent the vehicle from driving indefinitely away from the useful flat test region.

Do not use it as a navigation objective.

Phase 1 must not introduce:

- success-distance termination
- no-progress failure
- blocked-recovery termination
- stuck termination
- waypoint success
- recovery success

unless explicitly requested.

---

## Phase boundaries

The intended development sequence is:

### Phase 1 — Command following

Flat, high-traction terrain.

Validate:

- task registration
- observations
- continuous vehicle actions
- action rate limiting
- speed tracking
- heading tracking
- PPO learning behavior

Algorithm baseline:

**MLP-PPO**

### Phase 2 — Low-traction and proprioceptive estimation

After the Phase 1 baseline is stable, introduce:

- low-friction terrain
- traction variation
- physically observable traction-related signals
- proprioceptive estimation

Do not implement Phase 2 features while debugging the Phase 1 baseline.

### Phase 3 — Failure prediction and protective execution

Only after the earlier stages are stable, introduce:

- failure prediction
- protective execution
- intervention logic
- abort / recovery decisions where required by the research design

### Recurrent policies

Do not introduce GRU-PPO merely because the final research problem may require temporal memory.

Establish a stable MLP-PPO baseline first.

Use recurrence only when there is evidence that partial observability or temporal inference requires it.

---

## Reuse policy

Reuse existing code where it provides a clean implementation reference.

Potential reference sources include:

- `isaaclab_assets.CAR_CFG`
- `user/project/mdp/actions.py`
- existing `RateLimitedCarVWActionCfg`
- existing Car4WD command-following environment
- existing skrl PPO configuration
- generic `ManagerBasedRLEnvCfg` examples

Reuse should mean:

1. understand the original implementation,
2. extract the minimum useful mechanism,
3. rename it consistently,
4. remove legacy semantics that do not belong to the new task.

Do not blindly copy entire old modules.

---

## Legacy-code boundaries

### DreamerV3

Do not depend on:

```text
isaaclab_tasks.user.dreamerv3.*
```

for Phase 1 unless a specific reusable utility is explicitly justified.

Do not import:

- Dreamer runtime
- RSSM components
- Dreamer wrappers
- Dreamer-specific action terms
- TwoPointRecover logic
- BlockedRecovery logic
- medium-state logic

into the new CommandFollow task.

DreamerV3 remains available for later experiments but is not the architectural base of this new task.

### FailureAware project code

Do not carry the old FailureAware semantic model into Phase 1.

Do not expose or train against:

- `FailureAwareCommandCfg`
- `stuck_label`
- `slip_label`
- `abort_required_label`
- `front_traction_label`
- `rear_traction_label`

unless the user explicitly moves development into a later phase.

### Navigation-task logic

Do not introduce:

- waypoint navigation
- obstacle maps
- obstacle contact recovery
- height scanners
- waypoint pose commands
- navigation success distance
- obstacle-specific termination logic

into CommandFollow.

---

## Working rules

Treat this repository as the primary editable codebase.

Prefer task-specific changes over modifications to upstream Isaac Lab.

Do not modify the external Isaac Lab installation unless explicitly requested.

Before editing:

1. identify the current task and algorithm,
2. inspect the relevant implementation,
3. identify one concrete hypothesis or implementation goal,
4. inspect only the files necessary for that goal.

Make the smallest reasonable change that can test the hypothesis.

Do not change several unrelated environment or RL mechanisms in the same experiment.

After editing, inspect:

```bash
git diff
```

Do not refactor unrelated code while implementing a focused research change.

---

## Debugging priorities

For environment or training failures, first determine which layer contains the problem.

Recommended order:

### 1. Task registration

Check:

- Gym task registration
- Python package imports
- `env_cfg_entry_point`
- agent configuration entry point
- Train/Play task IDs

### 2. Scene and asset

Check:

- robot spawn
- joint names
- actuator configuration
- IMU prim path
- terrain
- reset state

### 3. Commands

Check:

- command shape
- resampling
- heading convention
- angle wrapping
- command ranges

### 4. Observations

Check:

- observation dimensions
- tensor shapes
- coordinate frames
- normalization/scaling
- NaN / Inf
- privileged-information leakage

### 5. Actions

Check:

- policy action shape
- bounding
- scaling
- rate limiting
- wheel target conversion
- steering target conversion
- actual executed action

### 6. Rewards

Check:

- sign
- magnitude
- scale
- frame convention
- reward domination
- unintended reward exploitation

### 7. Terminations and resets

Check:

- time-out semantics
- flip detection
- reset correctness
- unexpected truncations

### 8. Learning algorithm

Only after the environment interface is shown to be correct, inspect:

- PPO hyperparameters
- normalization
- learning rate
- rollout length
- minibatches
- entropy
- clipping
- value loss
- gradient stability

For non-PPO tasks, inspect the corresponding algorithm-specific components instead.

Do not immediately blame the RL algorithm for an environment-interface bug.

---

## Performance interpretation

Do not assume that higher raw episode return alone means that the controller improved.

For CommandFollow, evaluate relevant physical/task metrics such as:

- speed tracking error
- heading tracking error
- command-follow success
- action smoothness
- flip rate
- episode duration
- stability

When later research stages are added, introduce stage-specific metrics separately.

Do not use a future-stage metric to redefine success for Phase 1.

---

## Verification workflow

Use the lightest useful verification level.

Recommended order:

1. syntax / import checks
2. task registration check
3. environment creation
4. observation/action shape inspection
5. random-policy smoke test
6. short PPO training
7. play / checkpoint evaluation
8. longer baseline training only when justified

Do not immediately launch a long training run after modifying environment code.

A failed command must be diagnosed from its actual traceback or output before making another speculative change.

---

## Phase 1 acceptance checks

Before treating CommandFollow as a usable baseline, verify:

- the training task is discoverable by Gym
- the play task is discoverable by Gym
- the environment can be instantiated
- observation dimensions remain stable
- policy observations contain no unintended privileged information
- all observation tensors are finite
- action outputs are finite
- wheel velocity targets are finite
- steering targets are finite
- rate limiting behaves as configured
- random actions do not generate immediate NaN / Inf
- resets are stable
- PPO can start training
- speed-tracking reward shows meaningful learning trend
- heading-tracking reward shows meaningful learning trend
- learned behavior can be reproduced in play/evaluation

Do not move to Phase 2 solely because the environment runs.

The Phase 1 command-following baseline should first demonstrate meaningful learning.

---

## Experiment discipline

For every RL modification, record:

- research question or hypothesis
- task
- algorithm
- changed files
- changed parameters
- exact command
- log directory
- important metrics
- result
- conclusion

Use:

```text
agent/experiment_template.md
```

for experiment notes.

When comparing experiments, change as few variables as practical.

Do not claim that a modification improves performance without experimental evidence.

---

## Safety and repository hygiene

Do not run destructive commands unless explicitly requested, including:

```text
rm -rf
git reset --hard
git clean -fd
git push --force
```

Do not delete:

- checkpoints
- run directories
- W&B data
- useful experiment logs

Do not overwrite a known-good checkpoint.

Do not commit, push, merge, rebase, or rewrite Git history unless explicitly requested.

Do not install or upgrade:

- CUDA
- NVIDIA drivers
- Isaac Sim
- Isaac Lab
- PyTorch
- skrl
- core environment packages

merely to fix an application-level problem without first identifying and explaining the dependency issue.

Do not create a new Python environment unless explicitly requested.

---

## Agent helper scripts

Agent helper scripts may read machine-specific overrides from:

```text
agent/agent.env
```

`agent.env` contains local execution parameters only.

Do not use it to define permanent architectural assumptions about:

- the reinforcement-learning algorithm
- the research stage
- the project design

Because this repository may use several training stacks, helper scripts must not assume that every task uses DreamerV3.

If a helper script is specific to an algorithm, its name and configuration should make that dependency explicit.

Examples:

```text
agent_train_skrl_short.sh
agent_train_dreamer_short.sh
agent_eval_skrl.sh
```

rather than hiding different training systems behind an incorrect global default.

Never put:

- API keys
- W&B keys
- passwords
- access tokens
- other secrets

inside `AGENTS.md` or committed environment files.