import numpy as np

from awml_pred.datasets.transforms import MtrAgentEmbed, MtrPolylineEmbed
from awml_pred.typing import NDArray


def test_mtr_agent_embed(dummy_info) -> None:
    info, param = dummy_info
    transform = MtrAgentEmbed()
    ret = transform(info)

    ret_dim = (param.agent_dim - 2) + (param.num_past + 1) + (len(param.agent_types) + 2) + 4
    agent_past: NDArray = ret["agent_past"]
    agent_past_mask: NDArray = ret["agent_past_mask"]
    agent_past_pos: NDArray = ret["agent_past_pos"]
    agent_last_pos: NDArray = ret["agent_last_pos"]

    num_target = param.num_agent if param.predict_all_agents else param.num_target
    assert agent_past.shape == (
        num_target,
        param.num_agent,
        param.num_past,
        ret_dim,
    ), "Expected shape (B, N, Tp, D+Tp+C+5)"
    assert agent_past_mask.shape == (
        num_target,
        param.num_agent,
        param.num_past,
    ), "Expected shape (B, N, Tp)"
    assert agent_past_pos.shape == (num_target, param.num_agent, param.num_past, 3)
    assert agent_last_pos.shape == (num_target, param.num_agent, 3)

    assert np.allclose(
        agent_past,
        np.array(
            [
                [
                    [
                        [
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            0.0,
                            0.0,
                            1.0,
                            0.0,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            1.3817732,
                            -0.30116862,
                            0.0,
                            0.0,
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            0.0,
                            0.0,
                            1.0,
                            0.0,
                            0.0,
                            1.0,
                            0.0,
                            1.0,
                            0.0,
                            1.0,
                            1.3817732,
                            -0.30116862,
                            0.0,
                            0.0,
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            0.0,
                            0.0,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            2.0,
                            0.0,
                            1.0,
                            1.3817732,
                            -0.30116862,
                            0.0,
                            0.0,
                        ],
                    ],
                    [
                        [
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            1.3817732,
                            -0.30116862,
                            0.0,
                            0.0,
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            0.0,
                            1.0,
                            0.0,
                            1.0,
                            0.0,
                            1.0,
                            1.3817732,
                            -0.30116862,
                            0.0,
                            0.0,
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            0.0,
                            0.0,
                            1.0,
                            2.0,
                            0.0,
                            1.0,
                            1.3817732,
                            -0.30116862,
                            0.0,
                            0.0,
                        ],
                    ],
                    [
                        [
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            1.0,
                            1.0,
                            0.0,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            1.3817732,
                            -0.30116862,
                            0.0,
                            0.0,
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            1.0,
                            1.0,
                            0.0,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            0.0,
                            1.0,
                            0.0,
                            1.0,
                            1.3817732,
                            -0.30116862,
                            0.0,
                            0.0,
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            1.0,
                            1.0,
                            0.0,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            2.0,
                            0.0,
                            1.0,
                            1.3817732,
                            -0.30116862,
                            0.0,
                            0.0,
                        ],
                    ],
                ],
            ],
            dtype=np.float32,
        ),
    )

    assert np.allclose(
        agent_past_mask,
        np.array(
            [
                [
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                ],
            ],
            dtype=np.bool8,
        ),
    )

    # agent past pos
    assert np.allclose(
        agent_past_pos,
        np.array(
            [
                [
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                ],
            ],
            dtype=np.float32,
        ),
    )

    # agent last pos
    np.allclose(
        agent_last_pos,
        np.array(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
            ],
            dtype=np.float32,
        ),
    )


def test_mtr_polyline_embed(dummy_info) -> None:
    info, _ = dummy_info
    transform = MtrPolylineEmbed()
    ret = transform(info)

    polylines: NDArray = ret["polylines"]
    polylines_mask: NDArray = ret["polylines_mask"]
    polylines_center: NDArray = ret["polylines_center"]

    # polylines
    np.allclose(
        polylines,
        np.array(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 18.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 18.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 18.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 18.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 18.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                ],
            ],
            dtype=np.float32,
        ),
    )

    # polylines mask
    np.allclose(
        polylines_mask,
        np.array(
            [
                [
                    [
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                    ],
                ],
            ],
            dtype=np.bool8,
        ),
    )

    # polylines center
    np.allclose(polylines_center, np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32))
