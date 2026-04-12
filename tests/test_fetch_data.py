import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FetchDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fetch_data = _load_module("fetch_data", ROOT / "scripts" / "fetch_data.py")

    def test_extract_duplicate_refs_supports_multiple_markers(self):
        refs = self.fetch_data.extract_duplicate_refs(
            "This looks like duplicate of #123. Also same as #456 and see duplicate #789."
        )
        self.assertEqual(refs, [123, 456, 789])

    def test_extract_duplicate_refs_dedupes_matches(self):
        refs = self.fetch_data.extract_duplicate_refs(
            "duplicate of #42\nsame as #42\nduplicates #42"
        )
        self.assertEqual(refs, [42])


if __name__ == "__main__":
    unittest.main()
