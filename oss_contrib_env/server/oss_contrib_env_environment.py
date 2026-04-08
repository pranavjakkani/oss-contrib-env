# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import random
import re
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment

try:
    from ..benchmark import load_benchmark
    from ..grading import score_episode
    from ..models import OSSAction, OSSObservation, OSSState
except (ImportError, ModuleNotFoundError):
    from benchmark import load_benchmark
    from grading import score_episode
    from models import OSSAction, OSSObservation, OSSState


TASK_ALIASES = {
    "easy": "triage",
    "medium": "duplicate",
    "hard": "patch_loc",
}
MAX_ATTEMPTS = 5
BENCHMARK = load_benchmark()


def _resolve_task_type(task_id: str) -> str:
    if task_id in BENCHMARK["episodes"]:
        return task_id
    return TASK_ALIASES.get(task_id, "triage")


def _default_episode(task_type: str) -> dict[str, Any]:
    return BENCHMARK["episodes"][task_type][0]


def _parse_action(response: str) -> dict[str, Any]:
    text = (response or "").strip()
    if not text:
        return {"kind": "submit", "payload": ""}

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and isinstance(parsed.get("command"), str):
            command = parsed["command"].strip().lower()
            if command == "inspect":
                return {"kind": "inspect", "target": str(parsed.get("target", "")).strip()}
            if command == "submit":
                return {"kind": "submit", "payload": str(parsed.get("payload", "")).strip()}
    except json.JSONDecodeError:
        pass

    lowered = text.lower()
    if lowered.startswith("inspect "):
        return {"kind": "inspect", "target": text.split(" ", 1)[1].strip()}
    if lowered.startswith("submit "):
        return {"kind": "submit", "payload": text.split(" ", 1)[1].strip()}
    return {"kind": "submit", "payload": text}


def _find_candidate(task_type: str, episode: dict[str, Any], target: str) -> dict[str, Any] | None:
    if task_type == "patch_loc":
        for candidate in episode["candidates"]:
            if candidate["path"] == target:
                return candidate
        return None

    matches = re.findall(r"\d+", target or "")
    if not matches:
        return None
    candidate_id = int(matches[0])
    for candidate in episode["candidates"]:
        if candidate["id"] == candidate_id:
            return candidate
    return None


def _inspect_details(task_type: str, episode: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    issue_text = episode["issue"]
    if task_type == "patch_loc":
        path = candidate["path"]
        overlap = sorted({
            token for token in re.findall(r"[a-z][a-z0-9_]+", issue_text.lower())
            if token in path.lower()
        })
        return {
            "target": path,
            "directory": "/".join(path.split("/")[:-1]),
            "filename": path.split("/")[-1],
            "issue_overlap_terms": overlap[:8],
        }

    summary = candidate.get("summary", "")
    overlap = sorted({
        token for token in re.findall(r"[a-z][a-z0-9_]+", issue_text.lower())
        if token in summary.lower() or token in candidate.get("title", "").lower()
    })
    return {
        "target": candidate["id"],
        "title": candidate.get("title", ""),
        "labels": candidate.get("labels", []),
        "path_hints": candidate.get("path_hints", []),
        "issue_overlap_terms": overlap[:8],
        "summary": summary[:240],
    }


_shared: dict[str, Any] = {
    "task_type": "triage",
    "current_episode": _default_episode("triage"),
    "attempts": 0,
    "max_attempts": MAX_ATTEMPTS,
    "done": False,
    "last_score": None,
    "last_event": "loaded",
    "inspected_targets": [],
    "last_inspection": None,
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
        _shared["last_score"] = None
        _shared["last_event"] = "loaded"
        _shared["inspected_targets"] = []
        _shared["last_inspection"] = None
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

        parsed_action = _parse_action(action.response)
        if parsed_action["kind"] == "inspect":
            candidate = _find_candidate(_shared["task_type"], _shared["current_episode"], parsed_action["target"])
            if candidate is None:
                _shared["last_event"] = "inspect_failed"
                _shared["last_score"] = {
                    "progress": 0.0,
                    "penalty": 0.01,
                    "metrics": {},
                    "malformed": True,
                    "prediction": parsed_action["target"],
                }
                _shared["done"] = attempts_remaining == 0
                return self._make_obs(
                    reward=-0.01,
                    test_output=f"Inspect target not found: {parsed_action['target']}",
                    attempts_remaining=attempts_remaining,
                )

            inspection = _inspect_details(_shared["task_type"], _shared["current_episode"], candidate)
            target_key = str(inspection["target"])
            if target_key not in _shared["inspected_targets"]:
                _shared["inspected_targets"].append(target_key)
            _shared["last_inspection"] = inspection
            _shared["last_event"] = "inspect"
            prior_progress = 0.0 if _shared["last_score"] is None else _shared["last_score"]["progress"]
            _shared["last_score"] = {
                "progress": prior_progress,
                "penalty": 0.0,
                "metrics": {},
                "malformed": False,
                "prediction": inspection["target"],
            }
            _shared["done"] = attempts_remaining == 0
            return self._make_obs(
                reward=0.0,
                test_output=f"Inspection: {json.dumps(inspection, sort_keys=True)}",
                attempts_remaining=attempts_remaining,
            )

        score = score_episode(
            _shared["task_type"],
            parsed_action["payload"],
            _shared["current_episode"],
            _shared["attempts"],
        )
        _shared["last_event"] = "submit"
        _shared["last_score"] = score
        _shared["done"] = score["done"] or attempts_remaining == 0
        feedback = (
            f"Progress={score['progress']:.4f} | "
            f"Penalty={score['penalty']:.4f} | "
            f"Reward={score['reward']:.4f} | "
            f"Malformed={str(score['malformed']).lower()}"
        )
        return self._make_obs(
            reward=score["reward"],
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
        score = _shared["last_score"] or {
            "progress": 0.0,
            "penalty": 0.0,
            "metrics": {},
            "malformed": False,
            "prediction": None,
        }
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
                "progress": score["progress"],
                "penalty": score["penalty"],
                "malformed": score["malformed"],
                "prediction": score["prediction"],
                "metrics": score["metrics"],
                "status": _shared["last_event"],
                "available_actions": [
                    "inspect <candidate-id-or-path>",
                    "submit <final-answer>",
                ],
                "inspected_targets": list(_shared["inspected_targets"]),
                "last_inspection": _shared["last_inspection"],
                "ground_truth_available": True,
            },
        )

    @property
    def state(self) -> OSSState:
        return _shared["state"]
