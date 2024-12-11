from awml_pred.datatype.base import BaseType, ContextType


class AgentType(BaseType):
    """A base enum of Agent."""

    @staticmethod
    def to_context(*, as_str: bool = False) -> ContextType | str:
        """Convert the enum member to `ContextType`.

        Args:
        ----
            as_str (bool, optional): Whether to return as str. Defaults to False.

        Returns:
        -------
            ContextType | str: Converted object.

        """
        ctx = ContextType.AGENT
        return ctx.value if as_str else ctx
