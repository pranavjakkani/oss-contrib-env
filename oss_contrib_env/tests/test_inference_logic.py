import sys
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from inference import MODEL_NAME, normalize_action


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


if __name__ == "__main__":
    unittest.main()
