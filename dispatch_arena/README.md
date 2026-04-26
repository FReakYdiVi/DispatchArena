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
Dispatch Arena is a server-authoritative delivery-dispatch environment for OpenEnv, RL training, replay, and demo UI.

The environment focuses on a real delivery-operations problem: a dispatcher must decide when to assign, when to wait, when to reposition, and which order to prioritize while restaurant prep, travel time, and deadline pressure evolve over time. The same simulator is packaged for local testing, public demos, and GRPO training with Hugging Face TRL.

## Quick Links

- Hugging Face Space: [Freakdivi/dispatch_arena_v0](https://huggingface.co/spaces/Freakdivi/dispatch_arena_v0)
- Colab notebook: [Open in Colab](https://colab.research.google.com/drive/1vosZpVZF8TEBVER9KlTqNNQ4FFt8QUGN#scrollTo=1VjLY-otISBL)
- Blog / writeup: [HF blog.md](https://huggingface.co/spaces/Freakdivi/dispatch_arena_v0/blob/main/blog.md)
- Source repository: [FReakYdiVi/DispatchArena](https://github.com/FReakYdiVi/DispatchArena)
- Local notebook in repo: [`dispatch_arena_colab.ipynb`](./dispatch_arena_colab.ipynb)
- Training evidence directory: [`outputs/`](./outputs/)

## Why This Environment

Most delivery systems do not fail because a courier cannot move from point A to point B. They fail because timing decisions are wrong: a courier is assigned too early and waits at the restaurant, another courier is overloaded while someone else idles, or a high-pressure order is not prioritized until it is already late.

Dispatch Arena turns that timing problem into an RL environment. The agent is not controlling a courier one meter at a time. It acts like a dispatcher reading a shift dashboard and issuing semantic decisions such as `assign`, `reposition`, `hold`, and `prioritize`.

## How It Works

The environment exposes OpenEnv-style endpoints like `/reset`, `/step`, `/state`, and `/summary`. Internally, each step advances a server-authoritative simulator that updates couriers, orders, prep uncertainty, deadlines, and reward bookkeeping.

Each observation contains the visible dispatch state: courier locations, order status, deadlines, travel times, and recent events. The environment also returns legal actions and a decomposed reward so training can distinguish between progress, delivery success, invalid actions, lateness, idle time, and fairness costs.

## What Happens In A Step

### Act 1: The Cold Start

An episode begins with a dashboard, not a route. The agent sees multiple couriers, multiple active orders, and hidden restaurant prep uncertainty. At the start of training, it often makes locally plausible but globally bad decisions: assigning too early, overusing `hold`, or sending the wrong courier to the wrong order. Those choices lead to waiting, lateness, and negative reward.

### Act 2: The First Timing Insight

Very quickly, the agent discovers that dispatch is about timing, not only distance. A courier that is geographically closest may still be the wrong choice if the food is not ready, another order is about to arrive nearby, or that courier is already the most loaded one. The first real improvement is usually not a flashy strategy. It is simply learning that waiting or repositioning can be better than a premature assignment.

### Act 3: Pressure Builds

As more orders overlap, the environment becomes a coordination problem. One courier can get overloaded while another stays idle. A near-deadline order may need to be prioritized even if another assignment looks cheaper in the moment. Traffic noise and hidden prep times mean the agent has to learn policies that are robust to uncertainty, not just greedy one-step wins.

### Act 4: The Learning Signal

After every tool call, the simulator returns the next visible state and reward feedback. GRPO compares multiple rollout candidates from the same scenario and pushes the model toward trajectories that lead to better shift-level outcomes. That makes the learning signal about whole dispatch behavior over time, not just one isolated action.

## Results

The current run demonstrates that the full OpenEnv -> tool calling -> simulator -> reward -> GRPO loop is working end to end. The training curve below comes from a real GRPO run, and the baseline comparison shows where the current policy stands relative to simple non-learned policies.

This short run does **not** yet beat the heuristic ceiling. That is still useful evidence: the environment is trainable, the reward loop is real, and the benchmark already makes it clear what stronger future runs need to improve.

![Dispatch Arena training reward curve](./blog_assets/training_reward_curve.png)

![Dispatch Arena trained vs baselines](./blog_assets/trained_vs_baseline.png)

## Repository Layout

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

## Local Setup

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
