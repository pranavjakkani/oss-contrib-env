import json
import sys
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from baseline_agent import (
    build_candidate_preview,
    choose_baseline_action,
    choose_duplicate_action,
    choose_patch_loc_action,
    choose_triage_action,
)


class BaselineAgentTests(unittest.TestCase):
    def test_triage_prefers_profile_overlap(self):
        observation = {
            "task_type": "triage",
            "issue": (
                "IssueTriage: choose the single best issue.\n\n"
                'Contributor profile: {"focus_labels": ["enhancement"], "focus_paths": ["src/datasets/audio.py"], '
                '"focus_terms": ["audio", "metadata", "column"]}'
            ),
            "candidates": [
                {"id": 1, "title": "Fix dataset card typo", "labels": ["documentation"], "summary": "README typo", "path_hints": ["README.md"]},
                {"id": 2, "title": "Audio metadata column support", "labels": ["enhancement"], "summary": "audio metadata column bug", "path_hints": ["src/datasets/audio.py"]},
            ],
        }
        self.assertEqual(choose_triage_action(observation), "2")

    def test_duplicate_returns_json_ids(self):
        observation = {
            "task_type": "duplicate",
            "issue": "DuplicateDetection issue about progress bar env var mismatch in tqdm and config.",
            "candidates": [
                {"id": 11, "title": "Progress bar env var mismatch", "labels": ["duplicate"], "summary": "config and tqdm env var mismatch"},
                {"id": 22, "title": "Image feature docs typo", "labels": [], "summary": "docs typo only"},
            ],
        }
        self.assertEqual(json.loads(choose_duplicate_action(observation)), [11])

    def test_patch_localizer_ranks_matching_path_first(self):
        observation = {
            "task_type": "patch_loc",
            "issue": "PatchLocalization for iterable dataset typo in src/datasets/iterable_dataset.py",
            "candidates": [
                {"path": "src/datasets/table.py"},
                {"path": "src/datasets/iterable_dataset.py"},
                {"path": "tests/test_iterable_dataset.py"},
            ],
        }
        action = json.loads(choose_patch_loc_action(observation))
        self.assertEqual(action[0], "src/datasets/iterable_dataset.py")

    def test_candidate_preview_is_task_aware(self):
        preview = build_candidate_preview(
            {"task_type": "patch_loc", "candidates": [{"path": "src/datasets/foo.py"}]}
        )
        self.assertIn("src/datasets/foo.py", preview)

    def test_choose_baseline_action_dispatches(self):
        observation = {
            "task_type": "patch_loc",
            "issue": "Need change in src/datasets/foo.py",
            "candidates": [{"path": "src/datasets/foo.py"}],
        }
        self.assertEqual(json.loads(choose_baseline_action(observation)), ["src/datasets/foo.py"])


if __name__ == "__main__":
    unittest.main()
