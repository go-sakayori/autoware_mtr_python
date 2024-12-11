from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import webcolors

from awml_pred.datatype import ContextType, LabelType

if TYPE_CHECKING:
    from typing_extensions import Self

__all__ = (
    "ColorSemantics",
    "ColorMap",
    "ContextColor",
    "get_semantics_color",
    "get_context_color",
    "get_label_color",
    "LabelColor",
)


class ColorSemantics(str, Enum):
    """Represent semantics of colorization."""

    LABEL = "LABEL"
    CONTEXT = "CONTEXT"
    INSTANCE = "INSTANCE"

    @classmethod
    def from_str(cls, name: str) -> Self:
        """Construct a enum member from its name.

        Args:
        ----
            name (str): Name of the member.

        Returns:
        -------
            ColorSemantics: Constructed member.

        """
        name = name.upper()
        assert name in cls.__members__, f"{name} is not in enum members of {cls.__name__}."
        return cls.__members__[name]


class ContextColor(str, Enum):
    """Enum to represent color of context in color code.

    Hex color code reference: https://htmlcolorcodes.com/color-names

    # Volatiles:
    ------------
    ## Global contexts:
    -------------------
        - `AGENT`: All agents. (SteelBlue [70, 130, 180])
        - `POLYLINE`: All polylines. (Silver [192, 192, 192])

    ## Local contexts:
    ------------------
        - `EGO`: Ego vehicle. (DarkCyan [0, 139, 139])
        - `FOCAL_AGENT`: Focal agents. (Coral [255, 127, 80])
        - `OTHER_AGENT`: Non-focal agents. (LightBlue [173, 216, 230])
        - `LANE`: Center lines. (LightGray [211, 211, 211])
        - `ROADLINE`: Road lines. (SlateGray [112, 128, 144])
        - `ROADEDGE`: Road edges. (DarkGray [168, 169, 169])
        - `CROSSWALK`: Crosswalks. (Gold [255, 195, 0])
        - `SIGNAL`: Traffic signals. (Violet [238, 130, 238])

    ## Catch all contexts:
    ----------------------
        - `UNKNOWN`: Catch the other contexts. (Khaki [240, 230, 140])
    """

    # Global context
    AGENT = "#4682B4"
    POLYLINE = "#C0C0C0"

    # Local context
    EGO = "#008B8B"
    FOCAL_AGENT = "#FF7F50"
    OTHER_AGENT = "#ADD8E6"

    LANE = "#D3D3D3"
    ROADLINE = "#708090"
    ROADEDGE = "#808080"
    CROSSWALK = "#FFC300"
    SIGNAL = "#EE82EE"

    # Catch all contexts
    UNKNOWN = "#F0E68C"

    @classmethod
    def from_context(cls, ctx: str | ContextType) -> ContextColor:
        """Return color of specified context in str.

        Args:
        ----
            ctx (str | ContextType): Type of context.

        Returns:
        -------
            ContextColor: Instance with specified name.

        """
        ctx: str = ctx.name if isinstance(ctx, ContextType) else ctx.upper()
        if ctx not in cls.__members__:
            msg = f"{ctx} is not in enum members of {cls.__name__}, UNKNOWN is used."
            logging.warning(msg)
            return cls.UNKNOWN
        else:
            return cls.__members__[ctx]


class LabelColor(str, Enum):
    """Color of agents in color code."""

    # Agent
    EGO = "#008B8B"
    VEHICLE = "#1E90FF"  # DodgerBlue: (30, 144, 255)
    LARGE_VEHICLE = "#6495ED"  # CornflowerBlue: (100, 149, 237)
    PEDESTRIAN = "#FF69B4"  # HotPink: (255, 105, 180)
    MOTORCYCLIST = "#800000"  # Maroon: (128, 0, 0)
    CYCLIST = "#9370DB"  # MediumPurple: (147, 112, 219)
    UNKNOWN = "#F0E68C"  # Khaki: (240, 230, 140)

    # Polyline
    # TODO: define label specific colors for polyline
    LANE = "#D3D3D3"
    ROADLINE = "#708090"
    ROADEDGE = "#808080"
    CROSSWALK = "#FFC300"
    SIGNAL = "#EE82EE"

    @classmethod
    def from_label(cls, label: str | LabelType) -> LabelColor:
        """Construct a enum member from `LabelType`.

        Args:
        ----
            label (str | LabelType): `LabelType` member or its name.

        Returns:
        -------
            LabelColor: Constructed member.

        """
        name: str = label.name if isinstance(label, LabelType) else label.upper()
        if name not in cls.__members__:
            msg = f"{name} is not in enum members of {cls.__name__}, UNKNOWN is used."
            logging.warning(msg)
            return cls.UNKNOWN
        else:
            return cls.__members__[name]


def get_context_color(ctx: str | ContextType) -> str:
    """Return color of specified context type.

    Args:
    ----
        ctx (str | ContextType): Type of context.

    Returns:
    -------
        str: Color code in str.

    """
    return ContextColor.from_context(ctx).value


def get_label_color(label: str | LabelType) -> str:
    """Return color of specified label type.

    Args:
    ----
        label (str | LabelType): Type of agent defined in `LabelType`.

    Returns:
    -------
        str: Color code in str.

    """
    return LabelColor.from_label(label).value


def get_semantics_color(level: str | ColorSemantics, name: str | ContextType | LabelType | None = None) -> str:
    """Return color code by specified level and type.

    Args:
    ----
        level (str | ColorSemantics): Colorize level.
        name (str | ContextType | LabelType | None): Type name required for only color level of CONTEXT or LABEL.
            For instance color level, None is fine.

    Returns:
    -------
        str: Color code in str.

    """
    if isinstance(level, str):
        level = ColorSemantics.from_str(level)

    if level == ColorSemantics.INSTANCE:
        return ColorMap.get_random_hex()
    elif level == ColorSemantics.CONTEXT:
        return get_context_color(name)
    elif level == ColorSemantics.LABEL:
        return get_label_color(name)
    else:
        msg = f"Unexpected colorize level: {level}"
        raise ValueError(msg)


class ColorMap:
    COLORS = (
        np.array(
            [
                (0.000, 0.447, 0.741),
                (0.850, 0.325, 0.098),
                (0.929, 0.694, 0.125),
                (0.494, 0.184, 0.556),
                (0.466, 0.674, 0.188),
                (0.301, 0.745, 0.933),
                (0.635, 0.078, 0.184),
                (0.300, 0.300, 0.300),
                (0.600, 0.600, 0.600),
                (1.000, 0.000, 0.000),
                (1.000, 0.500, 0.000),
                (0.749, 0.749, 0.000),
                (0.000, 1.000, 0.000),
                (0.000, 0.000, 1.000),
                (0.667, 0.000, 1.000),
                (0.333, 0.333, 0.000),
                (0.333, 0.667, 0.000),
                (0.333, 1.000, 0.000),
                (0.667, 0.333, 0.000),
                (0.667, 0.667, 0.000),
                (0.667, 1.000, 0.000),
                (1.000, 0.333, 0.000),
                (1.000, 0.667, 0.000),
                (1.000, 1.000, 0.000),
                (0.000, 0.333, 0.500),
                (0.000, 0.667, 0.500),
                (0.000, 1.000, 0.500),
                (0.333, 0.000, 0.500),
                (0.333, 0.333, 0.500),
                (0.333, 0.667, 0.500),
                (0.333, 1.000, 0.500),
                (0.667, 0.000, 0.500),
                (0.667, 0.333, 0.500),
                (0.667, 0.667, 0.500),
                (0.667, 1.000, 0.500),
                (1.000, 0.000, 0.500),
                (1.000, 0.333, 0.500),
                (1.000, 0.667, 0.500),
                (1.000, 1.000, 0.500),
                (0.000, 0.333, 1.000),
                (0.000, 0.667, 1.000),
                (0.000, 1.000, 1.000),
                (0.333, 0.000, 1.000),
                (0.333, 0.333, 1.000),
                (0.333, 0.667, 1.000),
                (0.333, 1.000, 1.000),
                (0.667, 0.000, 1.000),
                (0.667, 0.333, 1.000),
                (0.667, 0.667, 1.000),
                (0.667, 1.000, 1.000),
                (1.000, 0.000, 1.000),
                (1.000, 0.333, 1.000),
                (1.000, 0.667, 1.000),
                (0.333, 0.000, 0.000),
                (0.500, 0.000, 0.000),
                (0.667, 0.000, 0.000),
                (0.833, 0.000, 0.000),
                (1.000, 0.000, 0.000),
                (0.000, 0.167, 0.000),
                (0.000, 0.333, 0.000),
                (0.000, 0.500, 0.000),
                (0.000, 0.667, 0.000),
                (0.000, 0.833, 0.000),
                (0.000, 1.000, 0.000),
                (0.000, 0.000, 0.167),
                (0.000, 0.000, 0.333),
                (0.000, 0.000, 0.500),
                (0.000, 0.000, 0.667),
                (0.000, 0.000, 0.833),
                (0.000, 0.000, 1.000),
                (0.000, 0.000, 0.000),
                (0.143, 0.143, 0.143),
                (0.286, 0.286, 0.286),
                (0.429, 0.429, 0.429),
                (0.571, 0.571, 0.571),
                (0.714, 0.714, 0.714),
                (0.857, 0.857, 0.857),
                (0.000, 0.447, 0.741),
                (0.314, 0.717, 0.741),
                (0.50, 0.5, 0),
            ],
        )
        .astype(np.float32)
        .reshape(-1, 3)
    )

    @classmethod
    def get_random_rgb(cls, *, normalize: bool = False) -> tuple[int, int, int] | tuple[float, float, float]:
        """Return color in RGB randomly.

        Args:
        ----
            normalize (bool, optional): Whether to return normalized color. Defaults to False.

        Returns:
        -------
            tuple[int, int, int] | tuple[float, float, float]: RGB color.

        """
        rgb = np.random.rand(3)
        if normalize:
            return tuple(rgb.tolist())
        else:
            return tuple((rgb * 244).astype(np.uint8).tolist())

    @classmethod
    def get_rgb(cls, index: int, *, normalize: bool = False) -> tuple[int, int, int] | tuple[float, float, float]:
        """Return color in RGB chosen from `cls.COLORS`.

        Args:
        ----
            index (int): An index of COLORS. There are 80 colors defined as class variable.
                If the input index is over 80, the number of mod80 is used.
            normalize (bool, optional): Whether to return normalized color. Defaults to False.

        Returns:
        -------
            tuple[int, int, int] | tuple[float, float, float]: RGB color.

        """
        index: int = index % 80
        if normalize:
            return tuple(cls.COLORS[index].tolist())
        else:
            return tuple((cls.COLORS[index] * 255).astype(np.uint8).tolist())

    @classmethod
    def get_random_hex(cls) -> str:
        """Return color in hex randomly.

        Returns
        -------
            str: Hex color code.

        """
        rgb = tuple((np.random.rand(3) * 255).astype(np.uint8).tolist())
        try:
            return webcolors.rgb_to_hex(rgb)
        except Exception:  # noqa: BLE001
            return cls.__get_closest_hex(rgb)

    @classmethod
    def get_hex(cls, index: int) -> str:
        """Return color in hex chosen from `cls.COLORS`.

        Args:
        ----
            index (int): An index of COLORS. There are 80 colors defined as class variable.
                If the input index is over 80, the number of mod80 is used.

        Returns:
        -------
            str: Hex color code.

        """
        index: int = index % 80
        rgb = tuple((cls.COLORS[index] * 255).astype(np.uint8).tolist())
        try:
            return webcolors.rgb_to_hex(rgb)
        except Exception:  # noqa: BLE001
            return cls.__get_closest_hex(rgb)

    @classmethod
    def __get_closest_hex(cls, rgb: tuple[int, int, int]) -> str:
        """Return color in nearest hex.

        Args:
        ----
            rgb (tuple[int, int, int]): RGB color.

        Returns:
        -------
            str: Hex color code.

        """
        min_colors: dict[float, str] = {}

        for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - rgb[0]) ** 2
            gd = (g_c - rgb[1]) ** 2
            bd = (b_c - rgb[2]) ** 2
            min_colors[(rd + gd + bd)] = name

        return min_colors[min(min_colors.keys())]
