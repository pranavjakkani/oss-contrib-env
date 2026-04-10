import sys
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from inference import MODEL_NAME, bounded_episode_score, normalize_action


class InferenceLogicTests(unittest.TestCase):
    def test_default_model_uses_hf_router_openai_model(self):
        self.assertEqual(MODEL_NAME, "openai/gpt-oss-120b:cerebras")

    def test_repeated_inspect_falls_back_to_heuristic(self):
        observation = {
            "info": {
                "inspected_targets": ["6450"],
            }
        }
        action = normalize_action(observation, "inspect 6450", 'submit ["6450"]')
        self.assertEqual(action, 'submit ["6450"]')

    def test_new_inspect_is_preserved(self):
        observation = {
            "info": {
                "inspected_targets": ["6450"],
            }
        }
        action = normalize_action(observation, "inspect 6184", 'submit ["6450"]')
        self.assertEqual(action, "inspect 6184")

    def test_combined_inspect_submit_falls_back_to_heuristic(self):
        observation = {
            "info": {
                "inspected_targets": [],
            }
        }
        action = normalize_action(observation, "inspect 6450submit []", 'inspect 6450')
        self.assertEqual(action, "inspect 6450")

    def test_empty_duplicate_submission_falls_back_to_heuristic(self):
        observation = {
            "info": {
                "inspected_targets": [],
            }
        }
        action = normalize_action(observation, "submit []", 'inspect 6450')
        self.assertEqual(action, "inspect 6450")

    def test_bounded_episode_score_stays_inside_open_interval(self):
        self.assertEqual(bounded_episode_score([]), 0.0001)
        self.assertEqual(bounded_episode_score([0.0, 1.0]), 0.5)
        self.assertEqual(bounded_episode_score([1.0]), 0.9999)


if __name__ == "__main__":
    unittest.main()
