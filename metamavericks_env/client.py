# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Metamavericks Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import MetamavericksAction, MetamavericksObservation


class MetamavericksEnv(
    EnvClient[MetamavericksAction, MetamavericksObservation, State]
):
    """
    Client for the Metamavericks Env Environment.
    """

    def _step_payload(self, action: MetamavericksAction) -> Dict:
        """
        Convert MetamavericksAction to JSON payload for step message.
        """
        return {
            "commands": action.commands,
        }

    def _parse_result(self, payload: Dict) -> StepResult[MetamavericksObservation]:
        """
        Parse server response into StepResult[MetamavericksObservation].
        """
        obs_data = payload.get("observation", {})
        observation = MetamavericksObservation(
            features=obs_data.get("features", []),
            metadata=obs_data.get("metadata", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
