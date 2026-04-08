import json
import re
from typing import Any


MIN_REWARD = -0.2
MAX_REWARD = 1.0
MALFORMED_PENALTY = 0.05
WRONG_PENALTY = 0.02
REPEAT_PENALTY = 0.02


def clamp_reward(value: float) -> float:
    return round(max(MIN_REWARD, min(MAX_REWARD, value)), 4)


def _extract_ints(response: str) -> list[int]:
    return [int(match) for match in re.findall(r"\d+", response or "")]


def parse_triage_prediction(response: str, candidates: list[dict[str, Any]]) -> tuple[int | None, bool]:
    text = (response or "").strip()
    if not text:
        return None, True

    candidate_ids = [candidate["id"] for candidate in candidates]

    try:
        parsed = json.loads(text)
        if isinstance(parsed, int):
            numbers = [parsed]
        elif isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], int):
            numbers = parsed
        else:
            numbers = _extract_ints(text)
    except json.JSONDecodeError:
        numbers = _extract_ints(text)

    if not numbers:
        return None, True

    value = numbers[0]
    if value in candidate_ids:
        return value, False
    if 0 <= value < len(candidate_ids):
        return candidate_ids[value], False
    if 1 <= value <= len(candidate_ids):
        return candidate_ids[value - 1], False
    return None, True


def parse_duplicate_predictions(response: str) -> tuple[list[int], bool]:
    text = (response or "").strip()
    if not text:
        return [], False

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            values = [int(item) for item in parsed if isinstance(item, int)]
        elif isinstance(parsed, int):
            values = [parsed]
        else:
            values = _extract_ints(text)
    except json.JSONDecodeError:
        values = _extract_ints(text)

    if not values and text:
        return [], True

    deduped = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped, False


def parse_patch_predictions(response: str) -> tuple[list[str], bool]:
    text = (response or "").strip()
    if not text:
        return [], False

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            values = [str(item).strip() for item in parsed if str(item).strip()]
        elif isinstance(parsed, str):
            values = [parsed.strip()]
        else:
            values = []
    except json.JSONDecodeError:
        values = []
        for chunk in re.split(r"[\n,]+", text):
            chunk = chunk.strip().strip("\"'")
            if chunk:
                values.append(chunk)

    if not values and text:
        return [], True

    deduped = []
    seen = set()
    for value in values[:5]:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped, False


def _f1_score(predicted: set[int], truth: set[int]) -> tuple[float, float, float]:
    if not predicted and not truth:
        return 1.0, 1.0, 1.0
    if not predicted or not truth:
        return 0.0, 0.0, 0.0
    true_positive = len(predicted & truth)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(truth) if truth else 0.0
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return round(precision, 4), round(recall, 4), round(f1, 4)


def _wrong_action_penalty(progress: float, malformed: bool, attempt_number: int) -> float:
    if progress >= 1.0:
        return 0.0
    penalty = MALFORMED_PENALTY if malformed else 0.0
    if not malformed and progress == 0.0:
        penalty += WRONG_PENALTY
    if attempt_number > 1 and progress < 1.0:
        penalty += REPEAT_PENALTY * (attempt_number - 1)
    return penalty


def score_triage(response: str, episode: dict[str, Any], attempt_number: int) -> dict[str, Any]:
    selected_issue_id, malformed = parse_triage_prediction(response, episode["candidates"])
    ranking = episode["ground_truth"]["top_3_issue_ids"]
    progress = 0.0
    if selected_issue_id == ranking[0]:
        progress = 1.0
    elif selected_issue_id == ranking[1]:
        progress = 0.5
    elif selected_issue_id == ranking[2]:
        progress = 0.2
    penalty = _wrong_action_penalty(progress, malformed, attempt_number)
    reward = clamp_reward(progress - penalty)
    return {
        "progress": progress,
        "reward": reward,
        "penalty": round(penalty, 4),
        "malformed": malformed,
        "prediction": selected_issue_id,
        "done": progress >= 1.0,
        "metrics": {
            "selected_issue_id": selected_issue_id,
            "best_issue_id": ranking[0],
            "top_3_issue_ids": ranking,
        },
    }


def score_duplicate(response: str, episode: dict[str, Any], attempt_number: int) -> dict[str, Any]:
    predicted, malformed = parse_duplicate_predictions(response)
    truth = episode["ground_truth"]["duplicate_issue_ids"]
    precision, recall, f1 = _f1_score(set(predicted), set(truth))
    progress = f1
    penalty = _wrong_action_penalty(progress, malformed, attempt_number)
    reward = clamp_reward(progress - penalty)
    return {
        "progress": progress,
        "reward": reward,
        "penalty": round(penalty, 4),
        "malformed": malformed,
        "prediction": predicted,
        "done": progress >= 1.0,
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "truth_duplicate_issue_ids": truth,
        },
    }


def score_patch_loc(response: str, episode: dict[str, Any], attempt_number: int) -> dict[str, Any]:
    predicted, malformed = parse_patch_predictions(response)
    truth = episode["ground_truth"]["files"]
    reciprocal_rank = 0.0
    for index, path in enumerate(predicted, start=1):
        if path in truth:
            reciprocal_rank = round(1.0 / index, 4)
            break
    hits = len([path for path in predicted if path in truth])
    recall_at_5 = round(hits / len(truth), 4) if truth else 0.0
    progress = round(min(1.0, reciprocal_rank + (0.1 * recall_at_5)), 4)
    penalty = _wrong_action_penalty(progress, malformed, attempt_number)
    reward = clamp_reward(progress - penalty)
    return {
        "progress": progress,
        "reward": reward,
        "penalty": round(penalty, 4),
        "malformed": malformed,
        "prediction": predicted,
        "done": progress >= 1.0,
        "metrics": {
            "mrr": reciprocal_rank,
            "recall_at_5": recall_at_5,
            "truth_files": truth,
        },
    }


def score_episode(task_type: str, response: str, episode: dict[str, Any], attempt_number: int) -> dict[str, Any]:
    if task_type == "triage":
        return score_triage(response, episode, attempt_number)
    if task_type == "duplicate":
        return score_duplicate(response, episode, attempt_number)
    if task_type == "patch_loc":
        return score_patch_loc(response, episode, attempt_number)
    raise ValueError(f"Unsupported task_type: {task_type}")
