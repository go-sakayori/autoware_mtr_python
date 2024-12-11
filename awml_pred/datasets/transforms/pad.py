from __future__ import annotations

import numpy as np

from awml_pred.common import TRANSFORMS
from awml_pred.dataclass import AWMLAgentScenario

__all__ = ("PadAgent",)


@TRANSFORMS.register()
class PadAgent:
    """Pad agent trajectory by filling with 0.0.

    Required Keys:
    -------------
        scenario (AWMLAgentScenario): `AWMLAgentScenario` instance.

    Updated Keys:
    ------------
        scenario (AWMLAgentScenario): `AWMLAgentScenario` instance.
    """

    def __init__(self, pad_num_agent: int) -> None:
        """Construct a new object.

        Args:
        ----
            pad_num_agent (int): The number of padding agents. If the original trajectory is the shape of (N, T, D),
                after processing the shape will be (`pad_num_agent`, T, D).

        """
        self.pad_num_agent = pad_num_agent

    def __call__(self, info: dict) -> dict:
        """Run transformation.

        Args:
        ----
            info (dict): Source info.

        Returns:
        -------
            dict: Output info.

        """
        scenario: AWMLAgentScenario = info["scenario"]

        num_agent, num_timestamp, num_dim = scenario.trajectory.shape

        if self.pad_num_agent < num_agent:
            return info

        pad_trajectory = np.zeros((self.pad_num_agent, num_timestamp, num_dim))
        pad_trajectory[:num_agent] = scenario.trajectory.copy()

        pad_types = np.zeros(self.pad_num_agent, dtype=object)
        pad_types[:num_agent] = scenario.types.copy()

        pad_ids = np.zeros(self.pad_num_agent, dtype=np.int32)
        pad_ids[:num_agent] = scenario.ids.copy()

        target_info = {"indices": scenario.target_indices, "difficulties": scenario.target_difficulties}

        pad_scenario_dict = {
            "scenario_id": scenario.scenario_id,
            "ego_index": scenario.ego_index,
            "current_time_index": scenario.current_time_index,
            "timestamps": scenario.timestamps,
            "trajectory": pad_trajectory,
            "types": pad_types,
            "ids": pad_ids,
            "target_info": target_info,
        }

        info["scenario"] = AWMLAgentScenario.from_dict(pad_scenario_dict)

        return info
