# Codex + Isaac Lab / DreamerV3 agent helper kit

Place these files at the root of the RL repository.

Expected layout:

```text
<repo>/
├── AGENTS.md
├── agent/
│   ├── agent.env.example
│   ├── objectives.md
│   ├── known_issues.md
│   └── experiment_template.md
└── scripts/
    ├── agent_smoke_test.sh
    ├── agent_train_short.sh
    └── agent_eval.sh
```

## Setup

```bash
cp agent/agent.env.example agent/agent.env
nano agent/agent.env
chmod +x scripts/agent_*.sh
```

Optionally add the contents of `gitignore.additions` to your repository `.gitignore`.

## Usage

Smoke test:

```bash
./scripts/agent_smoke_test.sh
```

Short training:

```bash
./scripts/agent_train_short.sh
```

Pass extra arguments through to the Dreamer entry point:

```bash
./scripts/agent_train_short.sh --wandb --wandb_project dreamerv3-isaaclab
```

Evaluation:

```bash
CHECKPOINT=/path/to/checkpoint ./scripts/agent_eval.sh
```

The scripts deliberately use wall-clock guards by default so an autonomous coding agent cannot accidentally launch an unbounded training run. Override timeout values in `agent/agent.env` or per invocation.
