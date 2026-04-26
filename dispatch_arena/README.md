---
title: Dispatch Arena Environment Server
emoji: 🚚
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Dispatch Arena

Dispatch Arena is one server-authoritative delivery-dispatch environment package for RL training, replay, and demo UI.

Dispatch Arena focuses on a real delivery-operations question: how should a dispatcher assign, delay, or reprioritize orders when restaurant prep, travel time, and deadline pressure evolve over time? The environment is built for OpenEnv so the same simulator can be used for manual inspection, frontend demos, and reinforcement learning with tool-using agents.

## Quick Links

- Hugging Face Space: [Freakdivi/dispatch_arena_v0](https://huggingface.co/spaces/Freakdivi/dispatch_arena_v0)
- Training notebook: [`dispatch_arena_colab.ipynb`](./dispatch_arena_colab.ipynb)
- Writeup draft: [`blog_new.md`](./blog_new.md)
- Training evidence directory: [`outputs/`](./outputs/)

## Layout

```text
dispatch_arena/
  pyproject.toml
  openenv.yaml
  models.py
  client.py
  server/
    app.py
    api.py
    env.py
    rewards.py
    scenarios.py
    replay_store.py
    serializers.py
    metrics.py
    static/
  tests/
  docs/SPEC.md
  outputs/
```

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ./dispatch_arena
```

## Run Server

```bash
.venv/bin/python -m dispatch_arena.server.app
```

The FastAPI app exposes OpenEnv-compatible endpoints (`/reset`, `/step`, `/state`) and frontend API endpoints under `/api`.

## Run Tests

```bash
python -m unittest discover -s dispatch_arena/tests
```

## Mini Mode

Mini mode is the primary training environment:

- 1 courier, 1 order
- nodes: `hub`, `pickup`, `dropoff`
- actions: `wait`, `go_pickup`, `go_dropoff`, `pickup`, `dropoff`
- hidden state: exact `prep_remaining`
- visible state: order readiness/status, courier location/load, remaining ticks

## Normal Mode

Normal mode keeps the same simulator core and adds:

- 2-5 couriers
- 3-10 active orders
- pickup/dropoff nodes, prep uncertainty, deadlines
- centralized dispatcher actions: `assign`, `reposition`, `hold`, `prioritize`

## Reward Breakdown

Every step returns a decomposed reward:

| Component | Purpose |
|---|---|
| `step_cost` | Per-tick cost |
| `progress_reward` | Movement, pickup, assignment progress |
| `invalid_penalty` | Illegal action penalty |
| `success_reward` | Delivery reward |
| `timeout_penalty` | Hard episode timeout |
| `on_time_bonus` | Normal-mode SLA success |
| `late_penalty` | Late delivery |
| `idle_penalty` | Idle courier cost |
| `route_churn_penalty` | Reassignment/churn cost |
| `fairness_penalty` | Optional load imbalance cost |

## Client Example

```python
from dispatch_arena.client import DispatchArenaClient

client = DispatchArenaClient()
obs = client.reset(seed=7)

while not obs.done:
    if "pickup" in obs.legal_actions:
        action = "pickup"
    elif "dropoff" in obs.legal_actions:
        action = "dropoff"
    else:
        action = obs.legal_actions[0]
    obs = client.step(action)

print(obs.verifier_status, obs.reward_breakdown.to_dict())
```

Hidden state such as exact `prep_remaining` is never included in hidden-mode observations, public state, summaries, or replay payloads.
