import sys
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from grading import score_duplicate, score_patch_loc, score_triage


class GradingTests(unittest.TestCase):
    def test_triage_rewards_top_rank(self):
        episode = {
            "candidates": [{"id": 101}, {"id": 202}, {"id": 303}, {"id": 404}, {"id": 505}],
            "ground_truth": {"top_3_issue_ids": [303, 202, 101]},
        }
        score = score_triage("303", episode, attempt_number=1)
        self.assertEqual(score["progress"], 1.0)
        self.assertEqual(score["reward"], 1.0)
        self.assertFalse(score["malformed"])

    def test_triage_malformed_action_is_negative(self):
        episode = {
            "candidates": [{"id": 101}, {"id": 202}, {"id": 303}, {"id": 404}, {"id": 505}],
            "ground_truth": {"top_3_issue_ids": [303, 202, 101]},
        }
        score = score_triage("best issue please", episode, attempt_number=1)
        self.assertEqual(score["progress"], 0.0)
        self.assertEqual(score["reward"], -0.05)
        self.assertTrue(score["malformed"])

    def test_duplicate_uses_f1_progress(self):
        episode = {
            "ground_truth": {"duplicate_issue_ids": [10, 20]},
        }
        score = score_duplicate("[10]", episode, attempt_number=1)
        self.assertEqual(score["progress"], 0.6667)
        self.assertEqual(score["reward"], 0.6667)
        self.assertEqual(score["metrics"]["precision"], 1.0)
        self.assertEqual(score["metrics"]["recall"], 0.5)

    def test_duplicate_accepts_string_ids_in_json(self):
        episode = {
            "ground_truth": {"duplicate_issue_ids": [6450]},
        }
        score = score_duplicate('["6450"]', episode, attempt_number=1)
        self.assertEqual(score["progress"], 1.0)
        self.assertEqual(score["reward"], 1.0)
        self.assertFalse(score["malformed"])

    def test_patch_localization_uses_mrr_plus_recall_bonus(self):
        episode = {
            "ground_truth": {"files": ["src/a.py", "src/b.py"]},
        }
        score = score_patch_loc('["src/x.py", "src/b.py"]', episode, attempt_number=1)
        self.assertEqual(score["metrics"]["mrr"], 0.5)
        self.assertEqual(score["metrics"]["recall_at_5"], 0.5)
        self.assertEqual(score["progress"], 0.55)
        self.assertEqual(score["reward"], 0.55)

    def test_repeated_wrong_attempt_adds_small_penalty(self):
        episode = {
            "ground_truth": {"files": ["src/a.py"]},
        }
        score = score_patch_loc('["src/z.py"]', episode, attempt_number=3)
        self.assertEqual(score["progress"], 0.0)
        self.assertEqual(score["reward"], -0.06)


if __name__ == "__main__":
    unittest.main()
