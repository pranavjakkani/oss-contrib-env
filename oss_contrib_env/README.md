---
title: OSS Contrib Env
emoji: 🧰
colorFrom: yellow
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - rl
  - open-source
---

# OSS Contrib Env

`oss-contrib-env` is an offline OpenEnv benchmark for training and evaluating LLM agents on realistic open-source contribution tasks. It replays cached GitHub history from `huggingface/datasets` and grades the agent on three tasks:

1. `triage` (`easy`): choose the best issue for a contributor profile.
2. `duplicate` (`medium`): return duplicate issue IDs from a candidate set.
3. `patch_loc` (`hard`): return a ranked list of likely file paths to edit.

The benchmark is offline at runtime. `data/snapshot.json` is the cached GitHub issue history, and `data/benchmark.json` is the curated benchmark built from that snapshot for task-ready episodes and ground truth.

The environment now supports interactive multi-step routes. An agent can inspect candidates or paths first, then submit a final answer later in the trajectory. This makes the environment closer to a long-running workflow instead of a single-shot classifier with retries.

## Reward Design

Rewards stay close to the task metric, with mild negative shaping for clearly wrong or malformed actions.

- `triage`: `1.0`, `0.5`, `0.2`, or `0.0` based on whether the chosen issue lands in the top 3.
- `duplicate`: F1 score over predicted duplicate issue IDs.
- `patch_loc`: `MRR + 0.1 * recall@5`, capped at `1.0`.
- Wrong-action shaping: malformed actions and repeated fully wrong attempts can push the final reward mildly negative, down to `-0.2`.
- Exploration actions such as `inspect ...` are allowed during the trajectory and keep the episode alive for later submission.
- `info["progress"]` always reports the unpenalized task progress so logs still show partial learning signal clearly.

## Observation Shape

The environment returns:

- `task_id`
- `task_type`
- `difficulty`
- `issue`
- `candidates`
- `attempts_remaining`
- `reward`
- `done`
- `info`

The `info` payload includes:

- `progress`
- `penalty`
- `malformed`
- `prediction`
- `metrics`

## Quick Demo

Run the server locally:

```bash
uvicorn server.app:app --reload
```

Reset a task:

```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"triage"}'
```

Submit an action:

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action":{"response":"7931"}}'
```

Example action formats:

- `triage`: `"submit 7931"`
- `duplicate`: `"submit [6450]"`
- `patch_loc`: `"submit [\"src/datasets/iterable_dataset.py\", \"tests/test_iterable_dataset.py\"]"`
- inspect route: `"inspect 7931"` or `"inspect src/datasets/iterable_dataset.py"`

## Benchmark Builder

Use the committed snapshot as-is:

```bash
python3 scripts/fetch_data.py
```

Refresh the raw snapshot from GitHub first, then rebuild the benchmark:

```bash
export GITHUB_TOKEN=your_token_here
python3 scripts/fetch_data.py --fetch-snapshot --target 1000
```

This writes:

- `data/snapshot.json`: cached raw issue history
- `data/benchmark.json`: curated episodes for `triage`, `duplicate`, and `patch_loc`

## Agent Baseline

`inference.py` runs a hybrid baseline:

- heuristic reranking for each task
- optional LLM refinement if `HF_TOKEN` is set and the OpenAI-compatible client is available
- fallback to heuristic-only execution otherwise
- route-aware behavior that can inspect first and submit later, producing multi-step trajectories in the logs

Run it against a local server:

```bash
python3 inference.py
```

Step logs include both final reward and `progress`.

## Tests

Run the offline test suite:

```bash
python3 -m unittest discover -s tests -v
```

The tests cover:

- benchmark data validation
- semantic environment reset and stepping
- reward math and negative-penalty clamping
- baseline heuristic action generation

## Project Layout

```text
oss_contrib_env/
├── benchmark.py
├── baseline_agent.py
├── grading.py
├── inference.py
├── client.py
├── models.py
├── openenv.yaml
├── data/
│   ├── snapshot.json
│   └── benchmark.json
├── scripts/
│   └── fetch_data.py
├── tests/
│   ├── test_baseline_agent.py
│   ├── test_benchmark_data.py
│   ├── test_environment_reset.py
│   └── test_grading.py
└── server/
    ├── app.py
    └── oss_contrib_env_environment.py
```
