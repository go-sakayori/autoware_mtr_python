from enum import Enum
from typing import Any


class BaseEnum(Enum):
    @classmethod
    def get(cls, value: Any) -> Any:
        """Get a enum member from the specified value.

        Args:
        ----
            value (Any): Any value of a member.

        Returns:
        -------
            Any: Constructed member.

        """
        for k in cls:
            if k.value == value:
                return k

        msg = f"Cannot find key by value: {value} in {cls}"
        raise KeyError(msg)


class IR(BaseEnum):
    """Represents intermediate representation enumerations."""

    ONNX = "onnx"
    DEFAULT = "default"


class Backend(BaseEnum):
    """Represents backend enumerations."""

    PYTORCH = "pytorch"
    TENSORRT = "tensorrt"
    ONNXRUNTIME = "onnxruntime"
    TVM = "tvm"
    DEFAULT = "default"
