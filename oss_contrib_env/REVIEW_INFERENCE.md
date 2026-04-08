# Claude Code Prompt — Review inference.py

## Context
This is a hackathon submission for Meta x PyTorch OpenEnv Round 1.
Deadline is tonight. Do a thorough review of inference.py against
the exact judging criteria. Be brutal — flag anything that will 
cause disqualification.

## What You Must Do
1. Read inference.py in full
2. Read server/oss_contrib_env_environment.py to understand the env API
3. Cross-check against every requirement below
4. Fix any issues you find directly — don't just flag them

---

## Mandatory Requirements (Disqualification if Missing)

### File Location
- [ ] File is named exactly `inference.py` (not inference_script.py, not run.py)
- [ ] File is in the PROJECT ROOT, not inside server/ or any subfolder

### Environment Variables — All 3 Must Exist
```python
API_BASE_URL = os.environ.get("API_BASE_URL", "...")
MODEL_NAME   = os.environ.get("MODEL_NAME", "...")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
```
- [ ] All 3 are read from environment variables using os.environ.get()
- [ ] None are hardcoded as plain strings

### LLM Client
- [ ] Uses OpenAI client ONLY — not requests, not httpx, not huggingface_hub
- [ ] Client is initialized like this:
```python
from openai import OpenAI
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
```

### Log Format — SACRED, Zero Tolerance for Deviation
Every episode MUST emit exactly these 3 log types in order.
Any missing field, wrong key name, or wrong format = wrong evaluation score.

```python
# At episode START:
print(json.dumps({
    "type": "[START]",
    "task_id": "<task id string>",
    "difficulty": "<easy|medium|hard>"
}))

# At each STEP:
print(json.dumps({
    "type": "[STEP]",
    "step": <int>,
    "action": "<what agent submitted>",
    "reward": <float 0.0-1.0>,
    "done": <bool>
}))

# At episode END:
print(json.dumps({
    "type": "[END]",
    "task_id": "<task id string>",
    "total_reward": <float>,
    "steps": <int>
}))
```

Check:
- [ ] [START] is printed before any step
- [ ] [STEP] is printed after EVERY step call
- [ ] [END] is printed after episode finishes
- [ ] All keys match exactly (case sensitive)
- [ ] All values are correct types (step is int, reward is float, done is bool)
- [ ] json.dumps() is used — not print(f"...") or plain strings

### Environment API Calls
- [ ] Uses requests library to hit the env (NOT openenv client)
- [ ] Reset: POST to {ENV_URL}/reset with body {"task_id": "<id>"}
- [ ] Step: POST to {ENV_URL}/step with body {"response": "<agent answer>"}
- [ ] ENV_URL defaults to "http://localhost:8000"
- [ ] Handles the response correctly: result["observation"], result["reward"], result["done"]

### Runs All 3 Tasks
- [ ] Runs easy, medium AND hard — not just one
- [ ] Each task is a separate episode with its own [START] and [END]

### Runtime Constraint
- [ ] Will complete in under 20 minutes on 2 vCPU / 8GB RAM
- [ ] No heavy model downloads at runtime
- [ ] max_tokens is reasonable (512 or less per call)
- [ ] No infinite loops — has a max_steps guard (3 is fine)

---

## Common Bugs to Check For

### Bug 1 — Wrong observation key
```python
# WRONG
obs = result["obs"]
obs = result["observation"]["observation"]

# CORRECT  
obs = result["observation"]
```

### Bug 2 — Reward outside 0.0-1.0
```python
# Check that reward is read as float, not left as string
reward = float(result.get("reward", 0.0))  # correct
```

### Bug 3 — step field is wrong type
```python
# WRONG — step must be int, not string
{"type": "[STEP]", "step": "1", ...}

# CORRECT
{"type": "[STEP]", "step": 1, ...}
```

### Bug 4 — done field is wrong type
```python
# WRONG — done must be bool
{"type": "[STEP]", "done": "true", ...}
{"type": "[STEP]", "done": 1, ...}

# CORRECT
{"type": "[STEP]", "done": True, ...}
```

### Bug 5 — Missing error handling
```python
# If the env returns a 500, inference.py should not crash
# Wrap requests calls in try/except
```

### Bug 6 — LLM system prompt missing task context
```python
# The system prompt must tell the LLM what kind of task it's doing
# Check that issue + code from observation are passed to the LLM
# NOT just a generic prompt
```

---

## After Fixing — Run This End to End Test

Make sure the server is running first:
```bash
uv run uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Then run inference with test token:
```bash
export HF_TOKEN="your_hf_token"
export ENV_URL="http://localhost:8000"
python inference.py
```

### Expected Output Shape
```
{"type": "[START]", "task_id": "easy_001", "difficulty": "easy"}
{"type": "[STEP]", "step": 0, "action": "...", "reward": 1.0, "done": true}
{"type": "[END]", "task_id": "easy_001", "total_reward": 1.0, "steps": 1}
{"type": "[START]", "task_id": "medium_001", "difficulty": "medium"}
...
{"type": "[END]", "task_id": "medium_001", "total_reward": ..., "steps": ...}
{"type": "[START]", "task_id": "hard_001", "difficulty": "hard"}
...
{"type": "[END]", "task_id": "hard_001", "total_reward": ..., "steps": ...}
```

### What Passing Looks Like
- [ ] No Python errors or tracebacks
- [ ] Exactly 3 [START] blocks
- [ ] Exactly 3 [END] blocks
- [ ] All rewards are floats between 0.0 and 1.0
- [ ] Finishes in under 5 minutes locally

---

## If inference.py Has Major Issues

Rewrite it from scratch using this template:

```python
import os
import json
import requests
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://huggingface.co/api/inference-proxy/together")
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:8000")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = """You are an expert Python developer helping fix bugs in open source code.

For EASY tasks: identify which function has the bug and what type of error it is.
For MEDIUM tasks: write ONLY the corrected Python function, no explanation.
For HARD tasks: write the complete corrected file with ALL functions fixed.

Be precise and concise."""

def call_llm(issue: str, code: str) -> str:
    prompt = f"Issue:\n{issue}\n\nCode:\n{code}\n\nYour response:"
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error calling LLM: {e}"

def run_episode(task_id: str) -> float:
    # Reset
    try:
        reset_resp = requests.post(
            f"{ENV_URL}/reset",
            json={"task_id": task_id},
            timeout=30
        )
        result = reset_resp.json()
    except Exception as e:
        print(json.dumps({"type": "[ERROR]", "task_id": task_id, "error": str(e)}))
        return 0.0

    obs = result["observation"]

    print(json.dumps({
        "type": "[START]",
        "task_id": obs["task_id"],
        "difficulty": obs["difficulty"]
    }))

    total_reward = 0.0
    step = 0
    max_steps = 3

    while not result.get("done", False) and step < max_steps:
        # Call LLM
        action = call_llm(obs["issue"], obs["code"])

        # Step env
        try:
            step_resp = requests.post(
                f"{ENV_URL}/step",
                json={"response": action},
                timeout=30
            )
            result = step_resp.json()
        except Exception as e:
            print(json.dumps({"type": "[ERROR]", "step": step, "error": str(e)}))
            break

        obs = result["observation"]
        reward = float(result.get("reward", 0.0))
        done = bool(result.get("done", False))
        total_reward = reward

        print(json.dumps({
            "type": "[STEP]",
            "step": step,
            "action": action[:200],
            "reward": reward,
            "done": done
        }))

        step += 1

    print(json.dumps({
        "type": "[END]",
        "task_id": task_id,
        "total_reward": total_reward,
        "steps": step
    }))

    return total_reward

if __name__ == "__main__":
    tasks = ["easy", "medium", "hard"]
    scores = {}
    for task in tasks:
        score = run_episode(task)
        scores[task] = score
    print(json.dumps({"type": "[SUMMARY]", "scores": scores}))
```

---

## Final Check After All Fixes
Run this and confirm zero errors:
```bash
python -c "import inference; print('imports OK')"
python inference.py 2>&1 | grep -E '\[START\]|\[STEP\]|\[END\]|\[ERROR\]'
```

All lines should show [START], [STEP], or [END] — zero [ERROR] lines.
