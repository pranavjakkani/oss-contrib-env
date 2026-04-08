import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _install_openenv_stubs() -> None:
    openenv_module = types.ModuleType("openenv")
    core_module = types.ModuleType("openenv.core")
    env_server_module = types.ModuleType("openenv.core.env_server")
    interfaces_module = types.ModuleType("openenv.core.env_server.interfaces")
    types_module = types.ModuleType("openenv.core.env_server.types")

    class Environment:
        pass

    class _BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    interfaces_module.Environment = Environment
    types_module.Action = _BaseModel
    types_module.Observation = _BaseModel
    types_module.State = _BaseModel

    sys.modules["openenv"] = openenv_module
    sys.modules["openenv.core"] = core_module
    sys.modules["openenv.core.env_server"] = env_server_module
    sys.modules["openenv.core.env_server.interfaces"] = interfaces_module
    sys.modules["openenv.core.env_server.types"] = types_module


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class EnvironmentResetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_openenv_stubs()
        cls.benchmark = _load_module("benchmark", ROOT / "benchmark.py")
        cls.models = _load_module("models", ROOT / "models.py")
        cls.env_module = _load_module(
            "oss_contrib_env_environment",
            ROOT / "server" / "oss_contrib_env_environment.py",
        )

    def test_reset_uses_semantic_task_ids(self):
        env = self.env_module.OSSContribEnvironment()
        observation = env.reset(task_id="triage", seed=1)
        self.assertEqual(observation.task_type, "triage")
        self.assertEqual(observation.difficulty, "easy")
        self.assertGreater(len(observation.candidates), 0)
        self.assertEqual(observation.info["progress"], 0.0)
        self.assertEqual(env.state.current_task, "triage")

    def test_reset_supports_difficulty_aliases(self):
        env = self.env_module.OSSContribEnvironment()
        observation = env.reset(task_id="medium", seed=1)
        self.assertEqual(observation.task_type, "duplicate")
        self.assertEqual(observation.difficulty, "medium")

    def test_step_tracks_attempts_without_scoring_yet(self):
        env = self.env_module.OSSContribEnvironment()
        env.reset(task_id="patch_loc", seed=1)
        action = self.models.OSSAction(response="src/datasets/iterable_dataset.py")
        first = env.step(action)
        self.assertGreaterEqual(first.reward, -0.2)
        self.assertEqual(first.attempts_remaining, 2)
        self.assertFalse(first.done)
        self.assertEqual(first.info["status"], "graded")
        self.assertIn("progress", first.info)


if __name__ == "__main__":
    unittest.main()
