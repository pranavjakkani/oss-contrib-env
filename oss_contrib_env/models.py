# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from openenv.core.env_server.types import Action, Observation, State
from typing import Optional, Literal


class OSSAction(Action):
    """What the LLM agent submits"""
    response: str  # agent's answer (explanation or fixed code)


class OSSObservation(Observation):
    """What the agent sees each step"""
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    issue: str                        # GitHub issue description
    code: str                         # The buggy code
    test_output: Optional[str] = None # Feedback after step() calls
    attempts_remaining: int
    done: bool = False
    reward: float = 0.0


class OSSState(State):
    """Episode metadata"""
    episode_id: str
    step_count: int
    current_task: str
    difficulty: str
