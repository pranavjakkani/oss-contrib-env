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

`oss-contrib-env` is an offline [OpenEnv](https://github.com/facebookresearch/openenv) benchmark that trains and evaluates LLM agents on realistic open-source contribution workflows. It replays cached GitHub issue history from `huggingface/datasets` and grades the agent across three progressively harder tasks, each designed for **multi-step interactive trajectories** rather than single-shot classification.

## Why Multi-Step Trajectories Matter

Real open-source workflows are not one-shot decisions. A developer reads an issue, scans related tickets, inspects relevant files, then acts. This environment encodes that structure:

- **Two action routes at every step**: `inspect <target>` to gather evidence, or `submit <answer>` to commit.
- **Inspection costs a step** — the agent cannot inspect everything for free and must decide when it has gathered enough signal.
- **Repeated wrong submissions are penalized** — the agent must reason across steps, not brute-force retry.
- **Progress is partial** — partial credit on duplicate and patch-loc tasks keeps gradient signal flowing even when the agent is not yet perfect.

The result is a trajectory space where different inspection sequences lead to different outcomes, creating the multi-route structure needed for reinforcement learning and agent benchmarking.

## Task Tiers

| Task | Difficulty | Action | Grading | Max Steps |
|------|-----------|--------|---------|-----------|
| `triage` | easy | Submit the single best issue ID for a contributor profile | Top-3 partial credit (1.0 / 0.5 / 0.2) | 7 |
| `duplicate` | medium | Submit the issue ID(s) that duplicate the given issue | F1 score over predicted vs. ground-truth set | 7 |
| `patch_loc` | hard | Submit a ranked list of up to 5 file paths likely needing edits | MRR + 0.1 × recall@5, capped at 1.0 | 7 |

### Triage (`easy`)

The agent sees a contributor profile (labels, areas of interest, vocabulary) alongside 5 shuffled candidates. It must identify the best-fit issue. Inspect unlocks the full summary and path hints for individual candidates before committing.

### Duplicate Detection (`medium`)

The agent sees a new issue and a pool of 20 candidates. It must identify which candidate(s) are duplicates. Ground truth is derived from explicit `duplicate_of` references and label-backed similarity scoring. The scoring is F1 over the predicted set, so the agent is rewarded for precision as well as recall — submitting a large noisy list is penalized.

### Patch Localization (`hard`)

The agent sees an issue description and a candidate pool of ~12 file paths. It must return up to 5 paths ranked by likelihood of requiring edits. Scoring combines Mean Reciprocal Rank (MRR) for the top prediction and recall@5 across all 5 slots. Inspect reveals directory context and vocabulary overlap for individual paths.

## Reward Design

```
triage:     reward = progress - penalty
            progress ∈ {1.0, 0.5, 0.2, 0.0}  (top-1, top-2, top-3, miss)

duplicate:  reward = F1(predicted, truth) - penalty
            F1 = 2 * precision * recall / (precision + recall)

patch_loc:  reward = min(1.0, MRR + 0.1 * recall@5) - penalty
```

**Penalty shaping:**
- `malformed action`: −0.05 per step
- `fully wrong submission (progress=0)`: −0.02 per step
- `repeated identical wrong attempt`: −0.02 × (attempt_number − 1) cumulative
- Final reward is clamped to `[−0.2, 1.0]`

`info["progress"]` always reports the raw unpenalized task metric so training logs show clean learning signal.

## Observation Shape

Every `reset` and `step` response returns an `OSSObservation`:

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | str | Unique episode identifier |
| `task_type` | str | `"triage"`, `"duplicate"`, or `"patch_loc"` |
| `difficulty` | str | `"easy"`, `"medium"`, or `"hard"` |
| `issue` | str | The GitHub issue text the agent must act on |
| `candidates` | list | Candidate issues or file paths to choose from |
| `attempts_remaining` | int | Steps left before the episode ends |
| `reward` | float | Reward from the last action |
| `done` | bool | True when the episode has ended |
| `info` | dict | `progress`, `penalty`, `malformed`, `prediction`, `metrics`, `inspected_targets`, `last_inspection`, `available_actions` |

## Action Format

```
inspect issue                                  # overview of candidate IDs and focus terms
inspect <issue-id>                             # full details for a specific candidate issue
inspect <file-path>                            # directory context for a specific path (patch_loc)
submit <answer>                                # commit final answer and receive graded reward
```

Examples:

```
inspect 6450
inspect src/datasets/arrow_dataset.py
submit [6450]
submit ["src/datasets/arrow_dataset.py", "tests/test_arrow_dataset.py"]
submit 7931
```

## Trajectory Example

```
[RESET]  task=duplicate  candidates=20 issues shown
[STEP 1] inspect 6450   → labels=[duplicate, enhancement], overlap_ratio=0.41, summary=...
[STEP 2] submit [6450]  → F1=1.0, progress=1.0, done=true, reward=1.0
[END]    success=true   steps=2  score=1.0000
```

Agents that inspect first and submit with confidence in 2 steps score the same as agents that submit cold — but the inspection route is available when the candidate set is ambiguous, enabling strategies that trade one step for higher precision.

## Quick Start

**Start the server:**

```bash
uvicorn server.app:app --reload
```

**Reset a task episode:**

```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "duplicate"}'
```

**Take a step:**

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"response": "inspect 6450"}}'
```

**Submit a final answer:**

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"response": "submit [6450]"}}'
```

## Running the Baseline Agent

`inference.py` runs a hybrid agent (heuristic reranking + optional LLM refinement):

```bash
# Heuristic-only (no API key needed)
python3 inference.py

# LLM-assisted (requires HF_TOKEN with inference access)
export HF_TOKEN=hf_...
python3 inference.py
```

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV_URL` | `http://localhost:8000` | OpenEnv server URL |
| `MODEL_NAME` | `openai/gpt-oss-120b:cerebras` | Model for LLM route |
| `HF_TOKEN` | `` | HuggingFace token (enables LLM route) |
| `API_BASE_URL` | HF inference proxy | Override the OpenAI-compatible base URL |

Step logs report `action`, `reward`, `progress`, and `done` for every step, and a final `success` flag and `score` per episode.

## Benchmark Builder

The committed `data/benchmark.json` is ready to use. To refresh the raw snapshot from GitHub and rebuild:

```bash
export GITHUB_TOKEN=your_token_here
python3 scripts/fetch_data.py --fetch-snapshot --target 1000
```

To rebuild from an existing snapshot without re-fetching:

```bash
python3 scripts/fetch_data.py
```

This writes:

- `data/snapshot.json` — raw cached issue history from `huggingface/datasets`
- `data/benchmark.json` — curated episodes for `triage`, `duplicate`, and `patch_loc`

## Benchmark Stats

| Task | Episodes | Candidates per Episode |
|------|----------|----------------------|
| `triage` | 24 | 5 |
| `duplicate` | 12 | 20 |
| `patch_loc` | 24 | ~12 |

## Running Tests

```bash
python3 -m unittest discover -s tests -v
```

The test suite covers:

- Benchmark data validation (ground truth integrity, candidate membership)
- Environment reset and step semantics
- Reward math and penalty clamping
- Baseline heuristic action generation

## Project Layout

```
oss_contrib_env/
├── benchmark.py              # Episode builder: snapshot → benchmark.json
├── baseline_agent.py         # Heuristic agent (triage, duplicate, patch_loc)
├── grading.py                # Scoring functions (F1, MRR, partial credit)
├── inference.py              # End-to-end agent runner
├── client.py                 # OpenEnv HTTP client helpers
├── models.py                 # OSSAction, OSSObservation, OSSState types
├── openenv.yaml              # OpenEnv environment manifest
├── data/
│   ├── snapshot.json         # Cached GitHub issue history
│   └── benchmark.json        # Curated episodes with ground truth
├── scripts/
│   └── fetch_data.py         # Snapshot fetch + benchmark rebuild script
├── tests/
│   ├── test_baseline_agent.py
│   ├── test_benchmark_data.py
│   ├── test_environment_reset.py
│   └── test_grading.py
└── server/
    ├── app.py                # FastAPI server entry point
    └── oss_contrib_env_environment.py  # OpenEnv Environment implementation
```

## Design Notes

**Why offline?** The benchmark runs entirely from `data/benchmark.json` with no live GitHub API calls at inference time. This makes episodes reproducible, latency-free, and suitable for large-scale RL training loops.

**Why three task types?** The three tasks cover the full arc of an open-source contribution: deciding which issue to work on (triage), recognizing duplicate work (duplicate), and finding the right files to change (patch_loc). Together they form a curriculum from easy single-choice decisions to hard multi-file localization.

**Why F1 for duplicate?** Duplicate detection is inherently a set-matching problem. F1 penalizes both over-prediction (low precision) and under-prediction (low recall), which matches the real cost of wrong duplicate decisions in a project.

**Why MRR for patch_loc?** Getting the most relevant file ranked first matters more than covering every possible file. MRR rewards confident, precise top predictions while recall@5 provides a secondary signal for coverage.
