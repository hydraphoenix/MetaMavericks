# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Metamavericks Env Environment.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field, conlist
from typing import Dict, Union

class MetamavericksAction(Action):
    """Action for the Metamavericks Env environment - commands for agents."""
    commands: conlist(float, min_length=6, max_length=6) = Field(..., description="Action commands for 3 agents")

class MetamavericksObservation(Observation):
    """Observation from the Metamavericks Env environment."""
    features: conlist(float, min_length=48, max_length=48) = Field(..., description="Observation features")
    metadata: Dict[str, Union[float, int, str, bool]] = Field(default_factory=dict, description="Metadata")
