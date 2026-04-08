---
title: OSS Contrib Env
emoji: "🛠️"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - code-agents
  - debugging
---

# OSS Contrib Env

OSS Contrib Env is an OpenEnv environment that simulates a lightweight open source contribution workflow.  
An agent receives a bug report, reads buggy Python code, submits a fix or diagnosis, and gets reward feedback based on how correct the submission is.

This project was built for the Meta PyTorch x Scaler OpenEnv Hackathon Round 1, where the goal is to create a real-world environment with:

- typed action, observation, and state models
- `reset()`, `step()`, and `state()` support
- at least 3 graded tasks
- reward values in the range `0.0` to `1.0`
- a working baseline inference script
- a working Dockerfile and Hugging Face deployment

Live Space: [BhargaviThati/oss_contrib_env](https://huggingface.co/spaces/BhargaviThati/oss_contrib_env)

## Why This Environment

Many coding benchmarks focus only on final correctness. Real engineering work is more structured:

- understand an issue report
- inspect broken code
- produce the right format of response
- improve over multiple attempts
- learn from partial feedback

OSS Contrib Env turns that workflow into a compact environment that is easy to run, easy to grade, and meaningful for agent evaluation.

## Round 1 Scope

Round 1 focuses on a complete and deployable OpenEnv environment.

This repository includes:

- 3 tasks: `easy`, `medium`, `hard`
- task-specific graders
- partial reward signals
- HTTP endpoints for `reset`, `step`, and `state`
- Docker support
- Hugging Face Spaces deployment
- a reproducible `inference.py` baseline using the OpenAI client

## Tasks

### Easy
The agent must identify:

- which function contains the bug
- what type of error it is

Reward logic:

- `1.0` if both function and error type are correct
- `0.5` if only the function is correct
- `0.0` otherwise

### Medium
The agent must return only the corrected Python function for `reverse_string`.

Reward logic:

- score is based on test cases passed
- final reward includes an attempt penalty after repeated tries

### Hard
The agent must return the full corrected file content for multiple broken utility functions.

Reward logic:

- score is based on the full test suite
- final reward includes an attempt penalty after repeated tries

## Environment API

### Reset

```http
POST /reset
Content-Type: application/json
```

Example body:

```json
{"task_id":"easy"}
```

### Step

```http
POST /step
Content-Type: application/json
```

Example body:

```json
{
  "action": {
    "response": "The bug is in calculate_average and it is an off-by-one error."
  }
}
```

### State

```http
GET /state
```

## Typed Spaces

### Action

`OSSAction`

```python
response: str
```

The action is the agent submission. Depending on the task, this can be:

- a diagnosis
- a corrected function
- a corrected full file

### Observation

`OSSObservation`

```python
task_id: str
difficulty: Literal["easy", "medium", "hard"]
issue: str
code: str
test_output: Optional[str]
attempts_remaining: int
done: bool
reward: float
```

The observation contains the bug report, the buggy code, reward feedback, and remaining attempts.

### State

`OSSState`

```python
episode_id: str
step_count: int
current_task: str
difficulty: str
```

## Reward Design

Rewards are intentionally shaped to reflect real debugging progress:

- correct submissions receive `1.0`
- partially correct submissions can receive intermediate scores
- repeated attempts reduce reward through an attempt penalty
- all rewards are clamped to the required `0.0` to `1.0` range

This makes the environment more useful than a binary pass/fail benchmark.

## Project Structure

```text
oss_contrib_env/
├── README.md
├── openenv.yaml
├── pyproject.toml
├── uv.lock
├── __init__.py
├── client.py
├── inference.py
├── models.py
├── REVIEW_INFERENCE.md
└── server/
    ├── app.py
    ├── Dockerfile
    ├── __init__.py
    ├── oss_contrib_env_environment.py
    └── requirements.txt
```

## Local Setup

### 1. Install dependencies

```bash
cd /Users/bhargavi/oss-contrib-env/oss_contrib_env
uv sync
```

### 2. Run the server

```bash
uv run python -m server.app --port 8000
```

### 3. Test the API

Health:

```bash
curl -sS http://localhost:8000/health
```

Reset:

```bash
curl -sS -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"easy"}'
```

Step:

```bash
curl -sS -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action":{"response":"The bug is in calculate_average and it is an off-by-one error."}}'
```

## Baseline Inference Script

The baseline runner is:

- [inference.py](/Users/bhargavi/oss-contrib-env/oss_contrib_env/inference.py)

It:

- uses the OpenAI client for all LLM calls
- reads `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`, and `ENV_URL`
- runs all tasks: `easy`, `medium`, `hard`
- emits strict structured logs with `[START]`, `[STEP]`, and `[END]`
- stays within the expected runtime envelope for Round 1

### Environment Variables

```python
API_BASE_URL = os.environ.get("API_BASE_URL", "https://huggingface.co/api/inference-proxy/together")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")
```

### Run the baseline

Start the server in one terminal:

```bash
cd /Users/bhargavi/oss-contrib-env/oss_contrib_env
uv run python -m server.app --port 8000
```

Run inference in another terminal:

```bash
cd /Users/bhargavi/oss-contrib-env/oss_contrib_env
uv run --with openai python /Users/bhargavi/oss-contrib-env/oss_contrib_env/inference.py
```

## Docker

Build:

```bash
cd /Users/bhargavi/oss-contrib-env/oss_contrib_env
docker build -t oss-contrib-env -f server/Dockerfile .
```

Run:

```bash
docker run --rm -p 8000:8000 oss-contrib-env
```

The Dockerized server was validated with:

- `GET /health`
- `POST /reset`
- `POST /step`

## Hugging Face Deployment

This environment is deployed as a Hugging Face Space and can be pushed with:

```bash
cd /Users/bhargavi/oss-contrib-env/oss_contrib_env
openenv push
```

Deployed Space:

- [https://huggingface.co/spaces/BhargaviThati/oss_contrib_env](https://huggingface.co/spaces/BhargaviThati/oss_contrib_env)

## Round 2 Direction: MCP-Connected OSS Workflow

Round 1 proves that the environment, grader, baseline, Docker setup, and HF deployment all work.

For Round 2, the goal is to evolve this from a compact debugging benchmark into a richer agent workflow powered by MCP-connected tools.

### What MCP Adds

Model Context Protocol can let the agent interact with external tools and structured context instead of relying only on a static prompt.

Planned Round 2 MCP-connected capabilities:

- GitHub MCP
  - fetch real issues, PRs, comments, and repository context
  - ground tasks in live open source workflows
- Filesystem / codebase MCP
  - inspect project files directly
  - support multi-file debugging and repo-aware fixes
- Test runner MCP
  - execute tests, inspect failures, and retry with richer feedback
- Search / docs MCP
  - fetch framework or library documentation relevant to the task
- Patch / edit MCP
  - move from “submit an answer” to “propose or apply a patch”

### Round 2 Vision

The Round 2 version of this environment can simulate a fuller contribution loop:

1. read a real issue
2. inspect a repository through MCP tools
3. run tests
4. generate a patch
5. get graded on correctness, efficiency, and tool usage

That would move the benchmark from isolated bug fixing toward realistic agentic software engineering.

### Why This Matters

This gives us a strong bridge from Round 1 to Round 2:

- Round 1: OpenEnv compliance, baseline reproducibility, deployability
- Round 2: MCP-native agent workflows, richer tooling, more realistic contribution tasks

## Submission Checklist

This repository is structured to satisfy the key Round 1 expectations:

- working OpenEnv environment
- typed models
- `reset`, `step`, and `state` support
- minimum 3 graded tasks
- reward range `0.0` to `1.0`
- working Dockerfile
- working Hugging Face Space
- baseline `inference.py`
- reproducible local evaluation flow

## References

- Scaler Meta PyTorch Hackathon Dashboard: [https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/dashboard](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/dashboard)
- Hugging Face Space: [https://huggingface.co/spaces/BhargaviThati/oss_contrib_env](https://huggingface.co/spaces/BhargaviThati/oss_contrib_env)

