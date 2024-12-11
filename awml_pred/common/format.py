import pprint
from enum import Enum
from typing import Any


def format2str(obj: object, abbreviation: int | None = None, class_key: str | None = None) -> str:
    """Format any class objects to string.

    Args:
    ----
        obj (object): Any class objects.
        abbreviation (int | None, optional): If `len(list_object) > abbreviation` abbreviate the string. \
            Defaults to None.
        class_key (str | None, optional): Class key to convert to dict. Defaults to None.

    Returns:
    -------
        str: Formatted string.

    """
    formatted_obj = format_class(obj, abbreviation, class_key)
    return "\n" + pprint.pformat(formatted_obj, indent=1, width=120, depth=None, compact=True) + "\n"


def format_class(obj: object, abbreviation: int | None = None, class_key: str | None = None) -> Any:
    """Convert any class objects to be suitable for logging.

    Args:
    ----
        obj (object): Any class objects.
        abbreviation (int | None, optional): If `len(list_object) > abbreviation` abbreviate the string. \
            Defaults to None.
        class_key (str | None, optional): Class key to convert to dict. Defaults to None.

    Returns:
    -------
        Any: Converted object to be suitable for logging.

    """
    if isinstance(obj, dict):
        ret = {key: format_class(value, abbreviation, class_key) for key, value in obj.items()}
    elif isinstance(obj, Enum):
        ret = str(obj)
    elif hasattr(obj, "_ast"):
        ret = format_class(obj._ast(), abbreviation)  # noqa: SLF001
    elif hasattr(obj, "__iter__") and not isinstance(obj, str):
        if abbreviation is not None and len(obj) > abbreviation:
            ret = f" --- length of element {len(obj)} ---,"
        else:
            ret = [format_class(v, abbreviation, class_key) for v in obj]
    elif hasattr(obj, "__dict__"):
        data = {
            key: format_class(value, abbreviation, class_key)
            for key, value in obj.__dict__.items()
            if not callable(value) and not key.startswith("_")
        }
        if class_key is not None and hasattr(obj, "__class__"):
            data[class_key] = obj.__class__.__name__
        ret = data
    else:
        ret = obj

    return ret
