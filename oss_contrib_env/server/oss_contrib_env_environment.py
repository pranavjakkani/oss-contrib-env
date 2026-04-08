# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment

try:
    from ..benchmark import load_benchmark
    from ..models import OSSAction, OSSObservation, OSSState
except (ImportError, ModuleNotFoundError):
    from benchmark import load_benchmark
    from models import OSSAction, OSSObservation, OSSState


TASK_ALIASES = {
    "easy": "triage",
    "medium": "duplicate",
    "hard": "patch_loc",
}
MAX_ATTEMPTS = 3
BENCHMARK = load_benchmark()


def _resolve_task_type(task_id: str) -> str:
    if task_id in BENCHMARK["episodes"]:
        return task_id
    return TASK_ALIASES.get(task_id, "triage")


def _default_episode(task_type: str) -> dict[str, Any]:
    return BENCHMARK["episodes"][task_type][0]


_shared: dict[str, Any] = {
    "task_type": "triage",
    "current_episode": _default_episode("triage"),
    "attempts": 0,
    "max_attempts": MAX_ATTEMPTS,
    "done": False,
    "state": OSSState(
        episode_id="ep_00000",
        step_count=0,
        current_task="triage",
        difficulty="easy",
    ),
}


class OSSContribEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self):
        pass

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "triage",
        **kwargs: Any,
    ) -> OSSObservation:
        task_type = _resolve_task_type(task_id)
        rng = random.Random(seed)
        episode = rng.choice(BENCHMARK["episodes"][task_type])

        _shared["task_type"] = task_type
        _shared["current_episode"] = episode
        _shared["attempts"] = 0
        _shared["done"] = False
        _shared["state"] = OSSState(
            episode_id=episode_id or f"ep_{rng.randint(10000, 99999)}",
            step_count=0,
            current_task=task_type,
            difficulty=episode["difficulty"],
        )
        return self._make_obs(
            reward=0.0,
            test_output="Episode loaded from offline benchmark cache.",
            attempts_remaining=_shared["max_attempts"],
        )

    def step(
        self,
        action: OSSAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> OSSObservation:
        if _shared["done"]:
            return self._make_obs(
                reward=0.0,
                test_output="Episode already done.",
                attempts_remaining=0,
            )

        _shared["attempts"] += 1
        _shared["state"].step_count += 1
        attempts_remaining = max(0, _shared["max_attempts"] - _shared["attempts"])
        _shared["done"] = attempts_remaining == 0

        response = (action.response or "").strip()
        feedback = (
            f"Received action for {_shared['task_type']}. "
            f"Task-native scoring will be applied in the grading phase. "
            f"Submission length={len(response)}."
        )
        return self._make_obs(
            reward=0.0,
            test_output=feedback,
            attempts_remaining=attempts_remaining,
        )

    def _make_obs(
        self,
        reward: float,
        test_output: str,
        attempts_remaining: int = 0,
    ) -> OSSObservation:
        episode = _shared["current_episode"]
        return OSSObservation(
            task_id=episode["task_id"],
            task_type=episode["task_type"],
            difficulty=episode["difficulty"],
            issue=episode["issue"],
            candidates=episode["candidates"],
            code="",
            test_output=test_output,
            attempts_remaining=attempts_remaining,
            done=_shared["done"],
            reward=reward,
            info={
                "progress": 0.0,
                "status": "loaded" if not _shared["state"].step_count else "pending_grading",
                "ground_truth_available": True,
            },
        )

    @property
    def state(self) -> OSSState:
        return _shared["state"]
