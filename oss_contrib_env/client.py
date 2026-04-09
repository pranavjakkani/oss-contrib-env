# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Oss Contrib Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import OSSAction, OSSObservation


class OssContribEnv(
    EnvClient[OSSAction, OSSObservation, State]
):
    """
    Client for the Oss Contrib Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with OssContribEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(OSSAction(response="Hello!"))
        ...     print(result.observation.task_id)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = OssContribEnv.from_docker_image("oss_contrib_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(OSSAction(response="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: OSSAction) -> Dict:
        """
        Convert OSSAction to JSON payload for step message.

        Args:
            action: OSSAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "response": action.response,
        }

    def _parse_result(self, payload: Dict) -> StepResult[OSSObservation]:
        """
        Parse server response into StepResult[OSSObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with OSSObservation
        """
        obs_data = payload.get("observation", {})
        observation = OSSObservation(
            task_id=obs_data.get("task_id", ""),
            task_type=obs_data.get("task_type", "triage"),
            difficulty=obs_data.get("difficulty", "easy"),
            issue=obs_data.get("issue", ""),
            candidates=obs_data.get("candidates", []),
            code=obs_data.get("code", ""),
            test_output=obs_data.get("test_output"),
            attempts_remaining=obs_data.get("attempts_remaining", 0),
            done=obs_data.get("done", payload.get("done", False)),
            reward=obs_data.get("reward", payload.get("reward", 0.0)),
            info=obs_data.get("info", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
