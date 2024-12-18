from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from dataclasses import field
from typing import Sequence

import numpy as np

from autoware_mtr.dataclass.agent import AgentState
from awml_pred.ops import rotate_along_z

def get_relative_history(reference_state: AgentState, history : deque[AgentState]) -> deque[AgentState]:
    relative_history = history.copy()
    for i, state in enumerate(history):
        transformed_state_xyz = state.xyz - reference_state.xyz
        transformed_state_xyz[:2] = rotate_along_z(transformed_state_xyz[:2],-reference_state.yaw)
        transformed_state_yaw = state.yaw - reference_state.yaw
        transformed_vxy = rotate_along_z(state.vxy,state.yaw -reference_state.yaw)

        relative_timestamp = (state.timestamp - history[0].timestamp) / 1000.0
        relative_state = AgentState(uuid=state.uuid,timestamp=relative_timestamp,label_id=state.label_id,xyz=transformed_state_xyz,size=state.size,yaw=transformed_state_yaw,vxy=transformed_vxy.reshape((2,)),is_valid=state.is_valid)
        relative_history.append(relative_state)
    return relative_history
