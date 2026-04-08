import json
import re
from typing import Any


WORD_RE = re.compile(r"[a-z][a-z0-9_]{2,}")
STOPWORDS = {
    "about", "after", "against", "allow", "also", "been", "because", "before",
    "being", "between", "bug", "candidate", "choose", "data", "dataset",
    "datasets", "detection", "duplicate", "feature", "file", "files", "for",
    "from", "have", "issue", "list", "most", "path", "paths", "please",
    "ranked", "report", "return", "should", "support", "that", "the", "this",
    "triage", "with",
}


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in WORD_RE.findall((text or "").lower())
        if token not in STOPWORDS
    }


def _overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left), len(right))


def _extract_profile(issue_text: str) -> dict[str, Any]:
    marker = "Contributor profile:"
    if marker not in (issue_text or ""):
        return {}
    payload = issue_text.split(marker, 1)[1].strip()
    first_line = payload.splitlines()[0].strip()
    try:
        return json.loads(first_line)
    except json.JSONDecodeError:
        return {}


def _triage_score(candidate: dict[str, Any], profile: dict[str, Any]) -> float:
    label_overlap = _overlap(
        set(candidate.get("labels", [])),
        set(profile.get("focus_labels", [])),
    )
    path_overlap = _overlap(
        set(candidate.get("path_hints", [])),
        set(profile.get("focus_paths", [])),
    )
    summary_tokens = _tokenize(f"{candidate.get('title', '')} {candidate.get('summary', '')}")
    term_overlap = _overlap(summary_tokens, set(profile.get("focus_terms", [])))
    return round((0.4 * label_overlap) + (0.4 * term_overlap) + (0.2 * path_overlap), 4)


def choose_triage_action(observation: dict[str, Any]) -> str:
    profile = _extract_profile(observation.get("issue", ""))
    candidates = observation.get("candidates", [])
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -_triage_score(candidate, profile),
            candidate.get("id", 0),
        ),
    )
    if not ranked:
        return ""
    return str(ranked[0]["id"])


def _duplicate_score(observation: dict[str, Any], candidate: dict[str, Any]) -> float:
    issue_tokens = _tokenize(observation.get("issue", ""))
    candidate_tokens = _tokenize(f"{candidate.get('title', '')} {candidate.get('summary', '')}")
    label_bonus = 0.1 if "duplicate" in candidate.get("labels", []) else 0.0
    return round(_overlap(issue_tokens, candidate_tokens) + label_bonus, 4)


def choose_duplicate_action(observation: dict[str, Any]) -> str:
    candidates = observation.get("candidates", [])
    scored = sorted(
        (
            (_duplicate_score(observation, candidate), candidate["id"])
            for candidate in candidates
        ),
        key=lambda item: (-item[0], item[1]),
    )
    picks = [issue_id for score, issue_id in scored if score >= 0.18][:3]
    return json.dumps(picks)


def _path_score(issue_text: str, path: str) -> float:
    issue_tokens = _tokenize(issue_text)
    path_tokens = _tokenize(path.replace("/", " ").replace(".", " "))
    direct_bonus = 0.2 if path in issue_text else 0.0
    filename = path.split("/")[-1]
    filename_bonus = 0.15 if filename and filename in issue_text else 0.0
    return round(_overlap(issue_tokens, path_tokens) + direct_bonus + filename_bonus, 4)


def choose_patch_loc_action(observation: dict[str, Any]) -> str:
    issue_text = observation.get("issue", "")
    candidates = observation.get("candidates", [])
    ranked = sorted(
        (
            (_path_score(issue_text, candidate["path"]), candidate["path"])
            for candidate in candidates
        ),
        key=lambda item: (-item[0], item[1]),
    )
    top_paths = [path for _, path in ranked[:5]]
    return json.dumps(top_paths)


def choose_baseline_action(observation: dict[str, Any]) -> str:
    task_type = observation.get("task_type", "triage")
    if task_type == "triage":
        return choose_triage_action(observation)
    if task_type == "duplicate":
        return choose_duplicate_action(observation)
    if task_type == "patch_loc":
        return choose_patch_loc_action(observation)
    return ""


def build_candidate_preview(observation: dict[str, Any], limit: int = 5) -> str:
    task_type = observation.get("task_type", "triage")
    lines = []
    for candidate in observation.get("candidates", [])[:limit]:
        if task_type == "patch_loc":
            lines.append(f"- {candidate.get('path', '')}")
        else:
            lines.append(
                f"- id={candidate.get('id')} title={candidate.get('title', '')} "
                f"labels={candidate.get('labels', [])}"
            )
    return "\n".join(lines) if lines else "None"
