# scripts/fetch_data.py
# One-shot script to fetch 500 closed issues from huggingface/datasets
# Run once: python scripts/fetch_data.py
# Then commit data/snapshot.json — never run again

import json
import os
import time
import requests

# ── Config ────────────────────────────────────────────────────
TOKEN   = os.environ.get("GITHUB_TOKEN", "")
HEADERS = {"Authorization": f"token {TOKEN}",
           "Accept": "application/vnd.github+json"} if TOKEN else {}
BASE    = "https://api.github.com"
REPO    = "huggingface/datasets"
TARGET  = 500
OUT     = os.path.join(os.path.dirname(__file__), "..", "data", "snapshot.json")

# ── Helpers ───────────────────────────────────────────────────
def get(url, params=None, retries=3):
    """GET with retry and rate-limit awareness."""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=15)
            if r.status_code == 403:
                reset_time = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(reset_time - int(time.time()), 1)
                print(f"\n⚠️  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            if r.status_code == 200:
                time.sleep(0.4)  # polite delay
                return r.json()
        except requests.RequestException as e:
            print(f"\n⚠️  Request error: {e}. Retrying...")
            time.sleep(2)
    return None

def fetch_pr_files(pr_number):
    """Get list of files changed in a PR."""
    files = get(f"{BASE}/repos/{REPO}/pulls/{pr_number}/files")
    if isinstance(files, list):
        return [f["filename"] for f in files[:20]]  # cap at 20 files
    return []

def extract_duplicate_of(body: str):
    """Extract issue number from 'duplicate of #NNN' in issue body."""
    body_lower = body.lower()
    for marker in ["duplicate of #", "duplicates #", "dup of #"]:
        if marker in body_lower:
            try:
                after = body_lower.split(marker)[1]
                num_str = ""
                for ch in after:
                    if ch.isdigit():
                        num_str += ch
                    elif num_str:
                        break
                if num_str:
                    return int(num_str)
            except Exception:
                pass
    return None

def is_good_first_issue(labels):
    """Check if issue is tagged as beginner-friendly."""
    good_labels = {"good first issue", "good-first-issue",
                   "help wanted", "bug", "enhancement"}
    return any(l.lower() in good_labels for l in labels)

# ── Main Fetch ────────────────────────────────────────────────
def main():
    print(f"🚀 Fetching up to {TARGET} closed issues from {REPO}...")
    if not TOKEN:
        print("⚠️  No GITHUB_TOKEN set. Using unauthenticated (60 req/hr limit).")
        print("   Set it: export GITHUB_TOKEN=your_token_here\n")

    issues_raw = []
    page = 1

    while len(issues_raw) < TARGET:
        print(f"   Page {page} — collected {len(issues_raw)}/{TARGET}", end="\r")
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
        if not batch or not isinstance(batch, list) or len(batch) == 0:
            print(f"\n   No more results at page {page}.")
            break

        # GitHub issues endpoint returns PRs too — filter them out
        real_issues = [i for i in batch if "pull_request" not in i]
        issues_raw.extend(real_issues)
        page += 1

    issues_raw = issues_raw[:TARGET]
    print(f"\n✅ Fetched {len(issues_raw)} issues. Now enriching...")

    snapshot = []
    for idx, issue in enumerate(issues_raw):
        print(f"   Enriching {idx+1}/{len(issues_raw)}: #{issue['number']}", end="\r")

        body      = (issue.get("body") or "")[:600]
        labels    = [l["name"] for l in issue.get("labels", [])]
        dup_of    = extract_duplicate_of(body)

        # Fetch changed files for issues that have a linked PR
        pr_files  = []
        linked_pr = issue.get("pull_request")
        if linked_pr:
            pr_url   = linked_pr.get("url", "")
            pr_num   = int(pr_url.rstrip("/").split("/")[-1]) if pr_url else None
            if pr_num:
                pr_files = fetch_pr_files(pr_num)

        snapshot.append({
            # Identity
            "id":           issue["number"],
            "title":        issue["title"],
            "body":         body,
            "labels":       labels,
            "created_at":   issue["created_at"],
            "closed_at":    issue["closed_at"],
            # Task 1 ground truth — IssueTriage
            "is_good_first_issue": is_good_first_issue(labels),
            # Task 2 ground truth — DuplicateDetection
            "duplicate_of": dup_of,
            # Task 3 ground truth — PatchLocalization
            "pr_files":     pr_files,
        })

    # Save
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(snapshot, f, indent=2)

    # Stats
    with_prs   = sum(1 for i in snapshot if i["pr_files"])
    with_dups  = sum(1 for i in snapshot if i["duplicate_of"])
    good_first = sum(1 for i in snapshot if i["is_good_first_issue"])

    print(f"\n\n📊 Snapshot stats:")
    print(f"   Total issues:        {len(snapshot)}")
    print(f"   With PR files:       {with_prs}  (ground truth for PatchLocalization)")
    print(f"   Marked duplicates:   {with_dups} (ground truth for DuplicateDetection)")
    print(f"   Good first issues:   {good_first} (ground truth for IssueTriage)")
    print(f"\n💾 Saved to: {OUT}")
    print(f"✅ Done. Commit data/snapshot.json and never run this again.")

if __name__ == "__main__":
    main()
