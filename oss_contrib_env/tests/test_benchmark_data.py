import unittest
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from benchmark import (
    build_benchmark,
    build_duplicate_episodes,
    build_issue_features,
    load_snapshot,
    validate_benchmark,
)


class BenchmarkDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = load_snapshot()
        cls.benchmark = build_benchmark(cls.snapshot)

    def test_all_tasks_have_curated_episodes(self):
        episodes = self.benchmark["episodes"]
        self.assertGreater(len(episodes["triage"]), 0)
        self.assertGreater(len(episodes["duplicate"]), 0)
        self.assertGreater(len(episodes["patch_loc"]), 0)

    def test_validation_passes(self):
        validate_benchmark(self.benchmark)

    def test_duplicate_truth_is_inside_candidates(self):
        for episode in self.benchmark["episodes"]["duplicate"]:
            candidate_ids = {candidate["id"] for candidate in episode["candidates"]}
            truth_ids = set(episode["ground_truth"]["duplicate_issue_ids"])
            self.assertTrue(truth_ids.issubset(candidate_ids))

    def test_patch_truth_is_inside_candidates(self):
        for episode in self.benchmark["episodes"]["patch_loc"]:
            candidate_paths = {candidate["path"] for candidate in episode["candidates"]}
            truth_paths = set(episode["ground_truth"]["files"])
            self.assertTrue(truth_paths.issubset(candidate_paths))

    def test_duplicate_builder_uses_snapshot_duplicate_refs(self):
        rows = [
            {
                "id": 100,
                "title": "Tokenizer cache fails on remote datasets",
                "body": "Regression in tokenizer cache invalidation for remote datasets.",
                "labels": [],
                "created_at": "2026-01-01T00:00:00Z",
                "closed_at": "2026-01-02T00:00:00Z",
                "is_good_first_issue": False,
                "duplicate_of": 200,
                "duplicate_refs": [200],
                "pr_files": [],
            },
            {
                "id": 200,
                "title": "Remote dataset tokenizer cache invalidation regression",
                "body": "Tokenizer cache invalidation breaks on remote datasets after refresh.",
                "labels": [],
                "created_at": "2025-12-20T00:00:00Z",
                "closed_at": "2025-12-21T00:00:00Z",
                "is_good_first_issue": False,
                "duplicate_of": None,
                "duplicate_refs": [],
                "pr_files": [],
            },
            {
                "id": 300,
                "title": "Remote dataset auth header missing",
                "body": "Dataset download skips auth header for remote datasets in refresh flow.",
                "labels": [],
                "created_at": "2025-12-18T00:00:00Z",
                "closed_at": "2025-12-19T00:00:00Z",
                "is_good_first_issue": False,
                "duplicate_of": None,
                "duplicate_refs": [],
                "pr_files": [],
            },
            {
                "id": 400,
                "title": "Streaming tokenizer regression in map",
                "body": "Tokenizer cache interacts badly with map refresh in streaming mode.",
                "labels": [],
                "created_at": "2025-12-17T00:00:00Z",
                "closed_at": "2025-12-18T00:00:00Z",
                "is_good_first_issue": False,
                "duplicate_of": None,
                "duplicate_refs": [],
                "pr_files": [],
            },
            {
                "id": 500,
                "title": "Remote cache bug in iterable datasets",
                "body": "Iterable datasets lose cache tokens during remote refresh.",
                "labels": [],
                "created_at": "2025-12-16T00:00:00Z",
                "closed_at": "2025-12-17T00:00:00Z",
                "is_good_first_issue": False,
                "duplicate_of": None,
                "duplicate_refs": [],
                "pr_files": [],
            },
            {
                "id": 600,
                "title": "Cache invalidation for dataset viewer refresh",
                "body": "Dataset viewer refresh shows stale cache after tokenizer updates.",
                "labels": [],
                "created_at": "2025-12-15T00:00:00Z",
                "closed_at": "2025-12-16T00:00:00Z",
                "is_good_first_issue": False,
                "duplicate_of": None,
                "duplicate_refs": [],
                "pr_files": [],
            },
        ]

        episodes = build_duplicate_episodes(build_issue_features(rows), limit=12)
        seeded_episode = next(episode for episode in episodes if episode["task_id"] == "duplicate_100")

        self.assertEqual(seeded_episode["ground_truth"]["duplicate_issue_ids"], [200])
        candidate_ids = {candidate["id"] for candidate in seeded_episode["candidates"]}
        self.assertIn(200, candidate_ids)


if __name__ == "__main__":
    unittest.main()
