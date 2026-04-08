# Claude Code Session Prompt — OSS Contrib Env

## YOUR FIRST MESSAGE TO CLAUDE CODE (copy-paste this exactly)

---

I'm building a submission for the Meta x PyTorch OpenEnv Hackathon.
Read this entire prompt before writing any code.

## What This Project Is

An RL environment called `oss_contrib_env` where an LLM agent learns to
fix bugs in Python code — given a GitHub issue description and broken code.

Real-world utility: developer productivity.
Verifiable reward: tests pass or fail = binary truth with partial credit.

## Project Location

My scaffold is already generated at:
~/Desktop/oss-contrib-env/oss_contrib_env/

The scaffold created these files (DO NOT recreate them, just edit):
- models.py                              ← YOU WILL EDIT THIS
- server/oss_contrib_env_environment.py  ← YOU WILL EDIT THIS
- client.py                              ← scaffold handles, leave it
- server/main.py                         ← scaffold handles, leave it
- server/Dockerfile                      ← scaffold handles, leave it
- openenv.yaml                           ← YOU WILL EDIT THIS
- pyproject.toml                         ← leave it

## Step 1 — Edit models.py

Replace the contents of models.py with these typed Pydantic models.
Use the openenv base classes exactly as shown.

```python
from openenv import Action, Observation, State
from typing import Optional, Literal

class OSSAction(Action):
    """What the LLM agent submits"""
    response: str  # agent's answer (explanation or fixed code)

class OSSObservation(Observation):
    """What the agent sees each step"""
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    issue: str                        # GitHub issue description
    code: str                         # The buggy code
    test_output: Optional[str] = None # Feedback after step() calls
    attempts_remaining: int
    done: bool = False
    reward: float = 0.0

class OSSState(State):
    """Episode metadata"""
    episode_id: str
    step_count: int
    current_task: str
    difficulty: str
```

## Step 2 — Edit server/oss_contrib_env_environment.py

This is the core logic. Replace its contents with the full environment below.

### The 3 Tasks

**EASY** — Agent reads issue + code, identifies which function has the bug and why.
Action type: text explanation.
Grader: checks if response mentions correct function name + error type.

**MEDIUM** — Agent reads broken function + failing test output, writes fixed version.
Action type: Python code (just the function).
Grader: executes agent's code against hidden test cases.

**HARD** — Agent reads a file with multiple broken functions + full test suite failure.
Must patch all functions. Action type: full corrected file content.
Grader: runs complete test suite.

### Reward Formula

```
base_score = tests_passed / total_tests   # 0.0 to 1.0
attempt_penalty = (attempts - 1) * 0.1   # penalise slow solvers
reward = max(0.0, min(1.0, base_score - attempt_penalty))
```

### Full Environment Code

```python
import random
import subprocess
import tempfile
import os
import sys
from openenv import Environment
from models import OSSAction, OSSObservation, OSSState

# ─────────────────────────────────────────────
# TASK DEFINITIONS
# ─────────────────────────────────────────────

EASY_TASK = {
    "task_id": "easy_001",
    "difficulty": "easy",
    "issue": (
        "Bug Report: calculate_average() returns wrong result.\n"
        "Input: [1, 2, 3] → Expected: 2.0 → Got: 3.0\n"
        "Please identify which function has the bug and what type of error it is."
    ),
    "code": (
        "def calculate_average(nums):\n"
        "    return sum(nums) / len(nums) + 1  # bug here\n\n"
        "def format_output(result):\n"
        "    return f'Average: {result:.2f}'\n"
    ),
    "correct_function": "calculate_average",
    "correct_error": "off-by-one",
}

MEDIUM_TASK = {
    "task_id": "medium_001",
    "difficulty": "medium",
    "issue": (
        "Bug Report: reverse_string() fails for empty strings and single chars.\n"
        "Fix the function so all test cases pass.\n"
        "Return ONLY the corrected Python function, no explanation."
    ),
    "code": (
        "def reverse_string(s):\n"
        "    result = ''\n"
        "    for i in range(len(s)):  # bug: wrong range\n"
        "        result += s[i]\n"
        "    return result\n"
    ),
    "test_cases": [
        ("hello", "olleh"),
        ("", ""),
        ("a", "a"),
        ("ab", "ba"),
        ("racecar", "racecar"),
    ],
}

HARD_TASK = {
    "task_id": "hard_001",
    "difficulty": "hard",
    "issue": (
        "Bug Report: Multiple functions in utils.py are broken.\n"
        "Tests are failing. Fix ALL functions so the full test suite passes.\n"
        "Return the complete corrected file content."
    ),
    "code": (
        "def add(a, b):\n"
        "    return a - b  # bug: should be +\n\n"
        "def multiply(a, b):\n"
        "    return a + b  # bug: should be *\n\n"
        "def is_even(n):\n"
        "    return n % 2 == 1  # bug: should be == 0\n\n"
        "def clamp(value, min_val, max_val):\n"
        "    return max(min_val, value)  # bug: missing min with max_val\n"
    ),
    "test_cases": [
        ("add(2, 3)", 5),
        ("multiply(3, 4)", 12),
        ("is_even(4)", True),
        ("is_even(3)", False),
        ("clamp(10, 0, 5)", 5),
        ("clamp(-1, 0, 5)", 0),
    ],
}

TASKS = {
    "easy": EASY_TASK,
    "medium": MEDIUM_TASK,
    "hard": HARD_TASK,
}

# ─────────────────────────────────────────────
# GRADERS
# ─────────────────────────────────────────────

def grade_easy(response: str, task: dict) -> float:
    """Check if agent identified the correct function and error type."""
    response_lower = response.lower()
    has_function = task["correct_function"].lower() in response_lower
    has_error = task["correct_error"].lower() in response_lower
    if has_function and has_error:
        return 1.0
    elif has_function:
        return 0.5
    return 0.0

def grade_medium(response: str, task: dict) -> float:
    """Execute agent's code against test cases."""
    test_cases = task["test_cases"]
    passed = 0
    for input_val, expected in test_cases:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                             delete=False) as f:
                f.write(response + "\n")
                f.write(f"result = reverse_string({repr(input_val)})\n")
                f.write(f"assert result == {repr(expected)}, f'Got {{result}}'\n")
                fname = f.name
            result = subprocess.run(
                [sys.executable, fname],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                passed += 1
        except Exception:
            pass
        finally:
            try:
                os.unlink(fname)
            except Exception:
                pass
    return round(passed / len(test_cases), 4)

def grade_hard(response: str, task: dict) -> float:
    """Run full test suite against agent's submitted file."""
    test_cases = task["test_cases"]
    passed = 0
    for expr, expected in test_cases:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                             delete=False) as f:
                f.write(response + "\n")
                f.write(f"result = {expr}\n")
                f.write(f"assert result == {repr(expected)}, "
                        f"f'Got {{result}}, expected {repr(expected)}'\n")
                fname = f.name
            result = subprocess.run(
                [sys.executable, fname],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                passed += 1
        except Exception:
            pass
        finally:
            try:
                os.unlink(fname)
            except Exception:
                pass
    return round(passed / len(test_cases), 4)

GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}

# ─────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────

class OSSContribEnvironment(Environment):

    def __init__(self):
        self.current_task = None
        self.attempts = 0
        self.max_attempts = 3
        self.done = False
        self._state = None
        self.difficulty = "easy"

    def reset(self, task_id: str = "easy") -> OSSObservation:
        self.difficulty = task_id if task_id in TASKS else "easy"
        self.current_task = TASKS[self.difficulty]
        self.attempts = 0
        self.done = False
        self._state = OSSState(
            episode_id=f"ep_{random.randint(10000, 99999)}",
            step_count=0,
            current_task=self.current_task["task_id"],
            difficulty=self.difficulty,
        )
        return OSSObservation(
            task_id=self.current_task["task_id"],
            difficulty=self.difficulty,
            issue=self.current_task["issue"],
            code=self.current_task["code"],
            test_output=None,
            attempts_remaining=self.max_attempts,
            done=False,
            reward=0.0,
        )

    def step(self, action: OSSAction) -> OSSObservation:
        if self.done:
            return self._make_obs(reward=0.0, test_output="Episode already done.")

        self.attempts += 1
        self._state.step_count += 1

        grader = GRADERS[self.difficulty]
        base_score = grader(action.response, self.current_task)
        attempt_penalty = (self.attempts - 1) * 0.1
        reward = round(max(0.0, min(1.0, base_score - attempt_penalty)), 4)

        self.done = base_score == 1.0 or self.attempts >= self.max_attempts
        attempts_remaining = max(0, self.max_attempts - self.attempts)

        feedback = (
            f"Score: {base_score:.2f} | "
            f"Attempts used: {self.attempts}/{self.max_attempts} | "
            f"Final reward: {reward:.4f}"
        )
        return self._make_obs(reward=reward, test_output=feedback,
                              attempts_remaining=attempts_remaining)

    def _make_obs(self, reward: float, test_output: str,
                  attempts_remaining: int = 0) -> OSSObservation:
        return OSSObservation(
            task_id=self.current_task["task_id"],
            difficulty=self.difficulty,
            issue=self.current_task["issue"],
            code=self.current_task["code"],
            test_output=test_output,
            attempts_remaining=attempts_remaining,
            done=self.done,
            reward=reward,
        )

    @property
    def state(self) -> OSSState:
        return self._state
```

## Step 3 — Edit openenv.yaml

Replace with:

```yaml
name: oss_contrib_env
version: "0.1.0"
description: >
  An RL environment where an LLM agent learns to identify and fix bugs
  in Python code given a GitHub issue description and broken codebase.
  Simulates a real-world open source contribution workflow.

tasks:
  - id: easy
    description: "Identify the buggy function and error type from an issue"
    difficulty: easy
    max_steps: 3
    reward_range: [0.0, 1.0]

  - id: medium
    description: "Fix a broken Python function to pass unit tests"
    difficulty: medium
    max_steps: 3
    reward_range: [0.0, 1.0]

  - id: hard
    description: "Patch multiple broken functions to pass a full test suite"
    difficulty: hard
    max_steps: 3
    reward_range: [0.0, 1.0]

api:
  reset: /reset
  step: /step
  state: /state
```

## Step 4 — Verify it runs locally

After editing the files, run:

```bash
cd ~/Desktop/oss-contrib-env/oss_contrib_env
uv sync
uv run uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Then in a second terminal test all 3 tasks:

```bash
# Test reset
curl -s -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}' | python3 -m json.tool

# Test step
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"response": "The bug is in calculate_average, it has an off-by-one error"}' \
  | python3 -m json.tool

# Test medium
curl -s -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "medium"}' | python3 -m json.tool
```

## What Claude Code Must NOT Do

- Do not install new packages without asking
- Do not modify client.py or server/main.py
- Do not change pyproject.toml unless a missing import requires it
- Do not rename any files
- Do not create new files unless explicitly asked
- Reward must ALWAYS be between 0.0 and 1.0 — clamp everything
- Never skip the uv sync step before running

## If You Hit Import Errors

The openenv base classes may use slightly different import paths.
Check what the scaffold already imports at the top of the generated files
and match that pattern exactly.

## Success Looks Like

```
curl reset → returns JSON with task_id, issue, code
curl step  → returns JSON with reward (0.0-1.0), done, test_output
No crashes, no 500 errors
All 3 tasks (easy/medium/hard) independently resettable
```

Once all 3 curl commands return valid JSON with rewards, Steps 1-2 are done.
Tell me and we'll move to inference.py together.
