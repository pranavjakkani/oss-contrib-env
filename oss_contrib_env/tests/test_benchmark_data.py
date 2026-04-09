import unittest
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from benchmark import build_benchmark, load_snapshot, validate_benchmark


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


if __name__ == "__main__":
    unittest.main()
