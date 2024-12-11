from __future__ import annotations

from typing import TYPE_CHECKING

from awml_pred.common import SCENARIO_FILTERS, TRANSFORMS

if TYPE_CHECKING:
    from awml_pred.dataclass import AWMLAgentScenario


__all__ = ("FilterScenario", "FilterScenarioByType", "FilterScenarioByPastMask")


@TRANSFORMS.register()
class FilterScenario:
    """Filter agent scenario by any registered transforms.

    Required Keys:
    --------------
        scenario (AWMLAgentScenario): `AWMLAgentScenario` instance.

    Update Keys:
    ------------
        scenario (AWMLAgentScenario): `AWMLAgentScenario` instance.
    """

    def __init__(self, filters: list[dict]) -> None:
        """Load registered transforms.

        Args:
        ----
            filters (list[dict]): List of filter configurations.

        """
        self.filters = [SCENARIO_FILTERS.build(hook) for hook in filters]

    def __call__(self, info: dict) -> dict:
        """Run transformations.

        Args:
        ----
            info (dict): Source info.

        Returns:
        -------
            dict | None: Output info. If there is no corresponding targets returns `None`.

        """
        for t in self.filters:
            info = t(info)
        return info


@SCENARIO_FILTERS.register()
class FilterScenarioByType:
    """Filter agent scenario by type.

    Required Keys:
    --------------
        scenario (AWMLAgentScenario): `AWMLAgentScenario` instance.

    Update Keys:
    ------------
        scenario (AWMLAgentScenario): `AWMLAgentScenario` instance.
    """

    def __call__(self, info: dict) -> dict | None:
        """Run transformation.

        Args:
        ----
            info (dict): Source info.

        Returns:
        -------
            dict: Output info.

        """
        scenario: AWMLAgentScenario = info["scenario"]
        types: list[str] = info["agent_types"]
        info["scenario"] = scenario.filter_by_type(types)

        return info


@SCENARIO_FILTERS.register()
class FilterScenarioByPastMask:
    """Filter agent scenario by past mask.

    Required Keys:
    --------------
        scenario (AWMLAgentScenario): `AWMLAgentScenario` instance.

    Update Keys:
    ------------
        scenario (AWMLAgentScenario): `AWMLAgentScenario` instance.
    """

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
        info["scenario"] = scenario.filter_by_past_mask()

        return info
