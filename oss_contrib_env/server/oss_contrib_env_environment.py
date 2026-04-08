# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
import subprocess
import tempfile
import os
import sys
from typing import Optional, Any
from openenv.core.env_server.interfaces import Environment

try:
    from ..models import OSSAction, OSSObservation, OSSState
except (ImportError, ModuleNotFoundError):
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
        fname = None
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
            if fname:
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
        fname = None
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
            if fname:
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
# SHARED STATE (persists across HTTP requests)
# ─────────────────────────────────────────────

_shared: dict = {
    "difficulty": "easy",
    "current_task": TASKS["easy"],
    "attempts": 0,
    "max_attempts": 3,
    "done": False,
    "state": OSSState(
        episode_id="ep_00000",
        step_count=0,
        current_task=TASKS["easy"]["task_id"],
        difficulty="easy",
    ),
}

# ─────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────

class OSSContribEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self):
        pass

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None,
              task_id: str = "easy", **kwargs: Any) -> OSSObservation:
        difficulty = task_id if task_id in TASKS else "easy"
        task = TASKS[difficulty]
        _shared["difficulty"] = difficulty
        _shared["current_task"] = task
        _shared["attempts"] = 0
        _shared["done"] = False
        _shared["state"] = OSSState(
            episode_id=f"ep_{random.randint(10000, 99999)}",
            step_count=0,
            current_task=task["task_id"],
            difficulty=difficulty,
        )
        return OSSObservation(
            task_id=task["task_id"],
            difficulty=difficulty,
            issue=task["issue"],
            code=task["code"],
            test_output=None,
            attempts_remaining=_shared["max_attempts"],
            done=False,
            reward=0.0,
        )

    def step(self, action: OSSAction, timeout_s: Optional[float] = None,
             **kwargs: Any) -> OSSObservation:
        if _shared["done"]:
            return self._make_obs(reward=0.0, test_output="Episode already done.")

        _shared["attempts"] += 1
        _shared["state"].step_count += 1

        grader = GRADERS[_shared["difficulty"]]
        base_score = grader(action.response, _shared["current_task"])
        attempt_penalty = (_shared["attempts"] - 1) * 0.1
        reward = round(max(0.0, min(1.0, base_score - attempt_penalty)), 4)

        _shared["done"] = base_score == 1.0 or _shared["attempts"] >= _shared["max_attempts"]
        attempts_remaining = max(0, _shared["max_attempts"] - _shared["attempts"])

        feedback = (
            f"Score: {base_score:.2f} | "
            f"Attempts used: {_shared['attempts']}/{_shared['max_attempts']} | "
            f"Final reward: {reward:.4f}"
        )
        return self._make_obs(reward=reward, test_output=feedback,
                              attempts_remaining=attempts_remaining)

    def _make_obs(self, reward: float, test_output: str,
                  attempts_remaining: int = 0) -> OSSObservation:
        task = _shared["current_task"]
        return OSSObservation(
            task_id=task["task_id"],
            difficulty=_shared["difficulty"],
            issue=task["issue"],
            code=task["code"],
            test_output=test_output,
            attempts_remaining=attempts_remaining,
            done=_shared["done"],
            reward=reward,
        )

    @property
    def state(self) -> OSSState:
        return _shared["state"]
