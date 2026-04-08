import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parent / "data"
SNAPSHOT_PATH = DATA_DIR / "snapshot.json"
BENCHMARK_PATH = DATA_DIR / "benchmark.json"

TASK_TYPES = ("triage", "duplicate", "patch_loc")
PATH_PATTERN = re.compile(
    r"(?:src|tests|docs)/[A-Za-z0-9_./-]+\.(?:py|md|yaml|yml|json|txt)|README\.md|setup\.py|pyproject\.toml"
)
ISSUE_REF_PATTERN = re.compile(r"#(\d+)")
WORD_PATTERN = re.compile(r"[a-z][a-z0-9_]{2,}")
STOPWORDS = {
    "about", "after", "against", "allow", "also", "been", "because", "before",
    "being", "break", "broken", "bug", "call", "calls", "cannot", "code",
    "column", "columns", "could", "data", "dataset", "datasets", "docs",
    "does", "during", "error", "fails", "failure", "feature", "file", "from",
    "function", "have", "issue", "latest", "load", "loading", "more", "need",
    "other", "path", "please", "problem", "python", "raise", "release",
    "return", "should", "support", "test", "tests", "that", "there", "this",
    "using", "when", "with", "would",
}


def load_snapshot(path: Path = SNAPSHOT_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def load_benchmark(path: Path = BENCHMARK_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


def save_benchmark(benchmark: dict[str, Any], path: Path = BENCHMARK_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(benchmark, indent=2))


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def tokenize(text: str) -> set[str]:
    return {
        token for token in WORD_PATTERN.findall((text or "").lower())
        if token not in STOPWORDS
    }


def extract_path_hints(text: str) -> list[str]:
    return sorted(set(PATH_PATTERN.findall(text or "")))


def extract_duplicate_refs(text: str) -> list[int]:
    body = (text or "").lower()
    refs: set[int] = set()
    markers = (
        "duplicate of #",
        "duplicates #",
        "dup of #",
        "same as #",
        "see duplicate #",
    )
    for marker in markers:
        start = 0
        while True:
            index = body.find(marker, start)
            if index == -1:
                break
            suffix = body[index + len(marker):]
            digits = []
            for char in suffix:
                if char.isdigit():
                    digits.append(char)
                elif digits:
                    break
            if digits:
                refs.add(int("".join(digits)))
            start = index + len(marker)
    return sorted(refs)


def build_issue_features(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    features: dict[int, dict[str, Any]] = {}
    for row in rows:
        title = row.get("title", "")
        body = row.get("body", "")
        labels = sorted({label.lower() for label in row.get("labels", [])})
        text = normalize_text(f"{title}\n{body}")
        tokens = tokenize(f"{title} {body}")
        paths = extract_path_hints(text)
        duplicate_refs = extract_duplicate_refs(text)
        features[row["id"]] = {
            **row,
            "labels_norm": labels,
            "text": text,
            "tokens": tokens,
            "paths": paths,
            "duplicate_refs": duplicate_refs,
        }
    return features


def overlap_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    common = len(left & right)
    if common == 0:
        return 0.0
    return round(common / max(len(left), len(right)), 4)


def similarity_score(base: dict[str, Any], candidate: dict[str, Any]) -> float:
    label_score = overlap_score(set(base["labels_norm"]), set(candidate["labels_norm"]))
    token_score = overlap_score(set(base["tokens"]), set(candidate["tokens"]))
    path_score = overlap_score(set(base["paths"]), set(candidate["paths"]))
    return round((0.45 * token_score) + (0.35 * label_score) + (0.20 * path_score), 4)


def build_contributor_profile(issue: dict[str, Any]) -> dict[str, Any]:
    top_terms = sorted(issue["tokens"])[:8]
    return {
        "summary": (
            "Contributor profile derived from historically beginner-friendly issues "
            "with overlapping labels, repo areas, and bug vocabulary."
        ),
        "focus_labels": issue["labels_norm"][:5],
        "focus_paths": issue["paths"][:3],
        "focus_terms": top_terms,
    }


def build_triage_episodes(features: dict[int, dict[str, Any]], limit: int = 24) -> list[dict[str, Any]]:
    rng = random.Random(7)
    positives = [
        issue for issue in features.values()
        if issue.get("is_good_first_issue")
    ]
    negatives = [
        issue for issue in features.values()
        if not issue.get("is_good_first_issue")
    ]
    episodes: list[dict[str, Any]] = []
    for positive in positives:
        scored = [
            (similarity_score(positive, other), other)
            for other in negatives
            if other["id"] != positive["id"]
        ]
        scored = [item for item in scored if item[0] > 0]
        if len(scored) < 4:
            continue
        scored.sort(key=lambda item: (-item[0], item[1]["id"]))
        distractors = [item[1] for item in scored[:4]]
        candidate_rows = [positive, *distractors]
        rng.shuffle(candidate_rows)
        ranked_ids = [positive["id"]] + [issue["id"] for issue in distractors[:2]]
        episodes.append({
            "task_id": f"triage_{positive['id']}",
            "task_type": "triage",
            "difficulty": "easy",
            "issue": (
                "IssueTriage: choose the single best issue for the contributor profile.\n\n"
                f"Contributor profile: {json.dumps(build_contributor_profile(positive), sort_keys=True)}"
            ),
            "candidate_count": len(candidate_rows),
            "candidates": [
                {
                    "id": issue["id"],
                    "title": issue["title"],
                    "labels": issue["labels_norm"],
                    "created_at": issue.get("created_at"),
                    "summary": issue["text"][:420],
                    "path_hints": issue["paths"][:3],
                }
                for issue in candidate_rows
            ],
            "ground_truth": {
                "best_issue_id": positive["id"],
                "top_3_issue_ids": ranked_ids,
            },
            "metadata": {
                "source_issue_id": positive["id"],
                "profile_labels": positive["labels_norm"][:5],
            },
        })
        if len(episodes) >= limit:
            break
    return episodes


def infer_duplicate_truth(
    issue: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> list[int]:
    explicit_refs = [ref for ref in issue["duplicate_refs"] if any(item["id"] == ref for item in candidates)]
    if explicit_refs:
        return explicit_refs
    if "duplicate" in issue["labels_norm"]:
        scored = [
            (similarity_score(issue, candidate), candidate["id"])
            for candidate in candidates
            if candidate["id"] != issue["id"]
        ]
        scored = [item for item in scored if item[0] >= 0.12]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [issue_id for _, issue_id in scored[:1]]
    return []


def build_duplicate_episodes(features: dict[int, dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    seeds = [
        issue for issue in features.values()
        if issue["duplicate_refs"] or "duplicate" in issue["labels_norm"]
    ]
    episodes: list[dict[str, Any]] = []
    for issue in seeds:
        candidate_pool = sorted(
            (
                (similarity_score(issue, other), other)
                for other in features.values()
                if other["id"] != issue["id"]
            ),
            key=lambda item: (-item[0], item[1]["id"]),
        )
        candidates = [other for score, other in candidate_pool if score > 0][:20]
        if len(candidates) < 5:
            continue
        duplicate_ids = infer_duplicate_truth(issue, candidates)
        if not duplicate_ids:
            continue
        episodes.append({
            "task_id": f"duplicate_{issue['id']}",
            "task_type": "duplicate",
            "difficulty": "medium",
            "issue": (
                "DuplicateDetection: return duplicate issue IDs for the new issue, "
                "or an empty list if none apply.\n\n"
                f"New issue #{issue['id']}: {issue['title']}\n\n{issue['text'][:700]}"
            ),
            "candidates": [
                {
                    "id": candidate["id"],
                    "title": candidate["title"],
                    "labels": candidate["labels_norm"],
                    "summary": candidate["text"][:280],
                }
                for candidate in candidates
            ],
            "ground_truth": {
                "duplicate_issue_ids": duplicate_ids,
            },
            "metadata": {
                "source_issue_id": issue["id"],
                "label_marked_duplicate": "duplicate" in issue["labels_norm"],
                "explicit_duplicate_refs": issue["duplicate_refs"],
            },
        })
        if len(episodes) >= limit:
            break
    return episodes


def build_patch_episodes(features: dict[int, dict[str, Any]], limit: int = 24) -> list[dict[str, Any]]:
    rng = random.Random(11)
    all_paths = sorted({
        path
        for issue in features.values()
        for path in issue["paths"]
        if path.startswith(("src/", "tests/", "docs/")) or path in {"README.md", "setup.py", "pyproject.toml"}
    })
    episodes: list[dict[str, Any]] = []
    for issue in features.values():
        ground_truth = issue["paths"][:5]
        if not ground_truth:
            continue
        distractor_pool = [path for path in all_paths if path not in ground_truth]
        if len(distractor_pool) < 7:
            continue
        rng.shuffle(distractor_pool)
        candidate_paths = ground_truth + distractor_pool[:7]
        rng.shuffle(candidate_paths)
        episodes.append({
            "task_id": f"patch_loc_{issue['id']}",
            "task_type": "patch_loc",
            "difficulty": "hard",
            "issue": (
                "PatchLocalization: return a ranked list of up to 5 file paths most likely "
                "to require edits.\n\n"
                f"Issue #{issue['id']}: {issue['title']}\n\n{issue['text'][:900]}"
            ),
            "candidates": [
                {
                    "path": path,
                }
                for path in candidate_paths
            ],
            "ground_truth": {
                "files": ground_truth,
            },
            "metadata": {
                "source_issue_id": issue["id"],
            },
        })
        if len(episodes) >= limit:
            break
    return episodes


def benchmark_stats(benchmark: dict[str, Any]) -> dict[str, Any]:
    return {
        task_type: {
            "episodes": len(benchmark["episodes"].get(task_type, [])),
        }
        for task_type in TASK_TYPES
    }


def build_benchmark(rows: list[dict[str, Any]]) -> dict[str, Any]:
    features = build_issue_features(rows)
    benchmark = {
        "meta": {
            "source_repo": "huggingface/datasets",
            "source_snapshot_count": len(rows),
            "offline": True,
            "seed": 0,
        },
        "episodes": {
            "triage": build_triage_episodes(features),
            "duplicate": build_duplicate_episodes(features),
            "patch_loc": build_patch_episodes(features),
        },
    }
    validate_benchmark(benchmark)
    benchmark["meta"]["stats"] = benchmark_stats(benchmark)
    return benchmark


def validate_benchmark(benchmark: dict[str, Any]) -> None:
    if "episodes" not in benchmark:
        raise ValueError("benchmark is missing episodes")
    for task_type in TASK_TYPES:
        episodes = benchmark["episodes"].get(task_type)
        if not episodes:
            raise ValueError(f"benchmark is missing curated episodes for {task_type}")
        for episode in episodes:
            if episode.get("task_type") != task_type:
                raise ValueError(f"{episode.get('task_id')} has mismatched task_type")
            if not episode.get("task_id"):
                raise ValueError("episode is missing task_id")
            if not episode.get("issue"):
                raise ValueError(f"{episode['task_id']} is missing issue text")
            if not episode.get("candidates"):
                raise ValueError(f"{episode['task_id']} has no candidates")
            truth = episode.get("ground_truth", {})
            if task_type == "triage":
                best = truth.get("best_issue_id")
                top_3 = truth.get("top_3_issue_ids", [])
                candidate_ids = {candidate["id"] for candidate in episode["candidates"]}
                if best not in candidate_ids:
                    raise ValueError(f"{episode['task_id']} best issue is not in candidates")
                if len(top_3) != 3:
                    raise ValueError(f"{episode['task_id']} must define exactly 3 ranked issues")
            elif task_type == "duplicate":
                truth_ids = truth.get("duplicate_issue_ids", [])
                candidate_ids = {candidate["id"] for candidate in episode["candidates"]}
                if not truth_ids:
                    raise ValueError(f"{episode['task_id']} is missing duplicate truth IDs")
                if not set(truth_ids).issubset(candidate_ids):
                    raise ValueError(f"{episode['task_id']} duplicate truth is outside candidates")
            elif task_type == "patch_loc":
                paths = truth.get("files", [])
                candidate_paths = {candidate["path"] for candidate in episode["candidates"]}
                if not paths:
                    raise ValueError(f"{episode['task_id']} is missing patch truth files")
                if not set(paths).issubset(candidate_paths):
                    raise ValueError(f"{episode['task_id']} patch truth is outside candidates")


def build_and_save_benchmark(
    snapshot_path: Path = SNAPSHOT_PATH,
    benchmark_path: Path = BENCHMARK_PATH,
) -> dict[str, Any]:
    rows = load_snapshot(snapshot_path)
    benchmark = build_benchmark(rows)
    save_benchmark(benchmark, benchmark_path)
    return benchmark
