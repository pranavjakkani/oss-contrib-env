import argparse
import json
import os
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from benchmark import BENCHMARK_PATH, SNAPSHOT_PATH, build_and_save_benchmark


TOKEN = os.environ.get("GITHUB_TOKEN", "")
HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github+json",
} if TOKEN else {}
BASE = "https://api.github.com"
REPO = "huggingface/datasets"
TARGET = 500


def get(url, params=None, retries=3):
    """GET with retry and rate-limit awareness."""
    import requests

    for _ in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=15)
            if response.status_code == 403:
                reset_time = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(reset_time - int(time.time()), 1)
                print(f"\nRate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            if response.status_code == 200:
                time.sleep(0.2)
                return response.json()
        except requests.RequestException as error:
            print(f"\nRequest error: {error}. Retrying...")
            time.sleep(2)
    return None


def fetch_snapshot(target: int = TARGET) -> list[dict]:
    print(f"Fetching up to {target} closed issues from {REPO}...")
    if not TOKEN:
        print("No GITHUB_TOKEN set. Using unauthenticated mode.")

    issues_raw = []
    page = 1
    while len(issues_raw) < target:
        print(f"  Page {page} — collected {len(issues_raw)}/{target}", end="\r")
        batch = get(
            f"{BASE}/repos/{REPO}/issues",
            params={
                "state": "closed",
                "per_page": 50,
                "page": page,
                "sort": "created",
                "direction": "desc",
            },
        )
        if not batch or not isinstance(batch, list):
            break
        real_issues = [issue for issue in batch if "pull_request" not in issue]
        if not real_issues:
            break
        issues_raw.extend(real_issues)
        page += 1

    snapshot = []
    for issue in issues_raw[:target]:
        body = issue.get("body") or ""
        labels = [label["name"] for label in issue.get("labels", [])]
        snapshot.append({
            "id": issue["number"],
            "title": issue["title"],
            "body": body[:4000],
            "labels": labels,
            "created_at": issue["created_at"],
            "closed_at": issue["closed_at"],
            "is_good_first_issue": any(
                label.lower() in {"good first issue", "good-first-issue", "help wanted"}
                for label in labels
            ),
            "duplicate_of": None,
            "pr_files": [],
        })
    return snapshot


def save_snapshot(snapshot: list[dict], path: Path = SNAPSHOT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fetch-snapshot",
        action="store_true",
        help="Refresh data/snapshot.json from GitHub before building the benchmark.",
    )
    args = parser.parse_args()

    if args.fetch_snapshot or not SNAPSHOT_PATH.exists():
        snapshot = fetch_snapshot()
        save_snapshot(snapshot, SNAPSHOT_PATH)
        print(f"\nSaved raw snapshot to {SNAPSHOT_PATH}")
    else:
        print(f"Using existing snapshot: {SNAPSHOT_PATH}")

    benchmark = build_and_save_benchmark(SNAPSHOT_PATH, BENCHMARK_PATH)
    stats = benchmark["meta"]["stats"]
    print("\nBenchmark stats:")
    for task_type, task_stats in stats.items():
        print(f"  {task_type}: {task_stats['episodes']} episodes")
    print(f"\nSaved curated benchmark to {BENCHMARK_PATH}")


if __name__ == "__main__":
    main()
