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
MAX_ATTEMPTS_BY_TASK = {
    "triage": 5,
    "duplicate": 7,
    "patch_loc": 7,
}
BENCHMARK = load_benchmark()


def _resolve_task_type(task_id: str) -> str:
    if task_id in BENCHMARK["episodes"]:
        return task_id
    return TASK_ALIASES.get(task_id, "triage")


def _default_episode(task_type: str) -> dict[str, Any]:
    return BENCHMARK["episodes"][task_type][0]


def _max_attempts(task_type: str) -> int:
    return MAX_ATTEMPTS_BY_TASK.get(task_type, 5)


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


def _tokenize_for_overlap(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z][a-z0-9_]+", (text or "").lower())
        if len(token) >= 3
    }


def _overlap_terms(issue_text: str, candidate_text: str) -> list[str]:
    issue_tokens = _tokenize_for_overlap(issue_text)
    candidate_tokens = _tokenize_for_overlap(candidate_text)
    return sorted(issue_tokens & candidate_tokens)


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


def _inspect_issue_overview(task_type: str, episode: dict[str, Any]) -> dict[str, Any]:
    issue_text = episode["issue"]
    issue_terms = sorted(_tokenize_for_overlap(issue_text))[:12]
    overview = {
        "target": "issue",
        "task_type": task_type,
        "focus_terms": issue_terms,
        "candidate_count": len(episode["candidates"]),
    }
    if task_type == "patch_loc":
        candidate_paths = [candidate["path"] for candidate in episode["candidates"]]
        directories = sorted({
            "/".join(path.split("/")[:-1])
            for path in candidate_paths
            if "/" in path
        })
        overview["candidate_directories"] = directories[:5]
    else:
        overview["candidate_ids"] = [candidate["id"] for candidate in episode["candidates"][:8]]
    return overview


def _inspect_details(task_type: str, episode: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    issue_text = episode["issue"]
    if task_type == "patch_loc":
        path = candidate["path"]
        overlap = _overlap_terms(issue_text, path)
        directory = "/".join(path.split("/")[:-1])
        sibling_paths = [
            other["path"]
            for other in episode["candidates"]
            if other["path"] != path and directory and other["path"].startswith(f"{directory}/")
        ]
        extension = path.rsplit(".", 1)[-1] if "." in path else ""
        return {
            "target": path,
            "directory": directory,
            "filename": path.split("/")[-1],
            "extension": extension,
            "issue_overlap_terms": overlap[:8],
            "sibling_candidates": sibling_paths[:3],
            "rationale": (
                "Inspection highlights path segments and neighboring files that align "
                "with the issue vocabulary."
            ),
        }

    summary = candidate.get("summary", "")
    combined_text = f"{candidate.get('title', '')}\n{summary}"
    overlap = _overlap_terms(issue_text, combined_text)
    overlap_ratio = round(len(overlap) / max(len(_tokenize_for_overlap(issue_text)), 1), 4)
    rationale = "Text overlap suggests this candidate is relevant."
    if task_type == "duplicate":
        if "duplicate" in candidate.get("labels", []):
            rationale = "Candidate is label-marked duplicate and shares issue vocabulary."
        else:
            rationale = "Candidate shares issue vocabulary and is worth duplicate investigation."
    return {
        "target": candidate["id"],
        "title": candidate.get("title", ""),
        "labels": candidate.get("labels", []),
        "path_hints": candidate.get("path_hints", []),
        "issue_overlap_terms": overlap[:8],
        "overlap_ratio": overlap_ratio,
        "summary": summary[:400],
        "rationale": rationale,
    }


_shared: dict[str, Any] = {
    "task_type": "triage",
    "current_episode": _default_episode("triage"),
    "attempts": 0,
    "max_attempts": _max_attempts("triage"),
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
        _shared["max_attempts"] = _max_attempts(task_type)
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
            target = parsed_action["target"].strip().lower()
            if target in {"issue", "overview"}:
                inspection = _inspect_issue_overview(_shared["task_type"], _shared["current_episode"])
            else:
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
                    "inspect issue",
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
