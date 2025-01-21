from __future__ import annotations
from std_msgs.msg import Header
from rclpy.duration import Duration
from numpy.typing import NDArray
from geometry_msgs.msg import Pose
from autoware_mtr.dataclass.agent import OriginalInfo
# from autoware_perception_msgs.msg import PredictedPath
# from autoware_perception_msgs.msg import PredictedObjects
# from autoware_perception_msgs.msg import PredictedObject
from autoware_planning_msgs.msg import Trajectory, TrajectoryPoint


from collections import deque
from dataclasses import dataclass
from dataclasses import field
from typing import List, Sequence
from geometry_msgs.msg import Quaternion

import numpy as np

from autoware_mtr.dataclass.agent import AgentState
from awml_pred.ops import rotate_along_z
from tf_transformations import quaternion_from_euler


def get_relative_histories(reference_states: List[AgentState], histories: dict[str, AgentState]) -> List[deque[AgentState]]:
    relative_histories = []
    for reference_state in reference_states:
        for i, history in enumerate(histories.values()):
            # print("-------history-------", history)
            relative_history = get_relative_history(reference_state, history)
            # print("-------history-------", history)

        relative_histories.append(relative_history)

    return relative_histories


def get_relative_history(reference_state: AgentState, history: deque[AgentState]) -> deque[AgentState]:
    relative_history = history.copy()
    # print("-------relative history-------")
    for i, state in enumerate(history):
        # print(" id: ", state.uuid, " i: ", i)

        transformed_state_xyz = state.xyz - reference_state.xyz
        transformed_state_xyz[:2] = rotate_along_z(transformed_state_xyz[:2], -reference_state.yaw)
        transformed_state_yaw = state.yaw - reference_state.yaw
        transformed_vxy = rotate_along_z(state.vxy, state.yaw - reference_state.yaw)

        relative_timestamp = (state.timestamp - history[0].timestamp) / 1000.0
        relative_state = AgentState(uuid=state.uuid, timestamp=relative_timestamp, label_id=state.label_id, xyz=transformed_state_xyz,
                                    size=state.size, yaw=transformed_state_yaw, vxy=transformed_vxy.reshape((2,)), is_valid=state.is_valid)
        # print("relative state: ", relative_state)
        relative_history.append(relative_state)
    # print("-------relative history-------")
    return relative_history


__all__ = ("to_predicted_objects",)


def to_trajectory(
    header: Header,
    infos: Sequence[OriginalInfo],
    pred_scores: NDArray,
    pred_trajs: NDArray,
    score_threshold: float,
) -> List[Trajectory]:
    """Convert predictions to Trajectory msg.

    Args:
        header (Header): Header of the input message.
        infos (Sequence[OriginalInfo]): List of original message information.
        pred_scores (NDArray): Predicted score tensor in the shape of (N, M).
        pred_trajs (NDArray): Predicted trajectory tensor in the shape of (N, M, T, 4).
        score_threshold (float): Threshold value of score.

    Returns:
        Trajectory: Instanced msg.
    """
    output = []
    # convert each object
    for info, cur_scores, cur_trajs in zip(infos, pred_scores, pred_trajs, strict=True):
        target_trajs = _to_trajectories(info, cur_scores, cur_trajs, score_threshold)
        # get longest trajectory by measuring distance between waypoints
        longest_traj = max(target_trajs, key=lambda x: sum(np.linalg.norm(
            np.diff([[p.pose.position.x, p.pose.position.y] for p in x.points], axis=0), axis=1)))
        longest_traj.header = header
        output.append(longest_traj)
    return output


def _to_trajectories(
    info: OriginalInfo,
    pred_scores: NDArray,
    pred_trajs: NDArray,
    score_threshold: float,
) -> List[Trajectory]:
    """Convert prediction of a single object to Trajectory msg.

    Args:
        info (ObjectInfo): Object original info.
        pred_scores (NDArray): Predicted score in the shape of (M,).
        pred_trajs (NDArray): Predicted trajectory in the shape of (M, T, 4).
        score_threshold (float): Threshold value of score.

    Returns:
        Trajectory: Instanced msg.
    """
    output = []

    # # convert each mode
    for cur_score, cur_traj in zip(pred_scores, pred_trajs, strict=True):
        if cur_score < score_threshold:
            continue
        cur_mode_path = _to_traj(info, cur_traj)
        output.append(cur_mode_path)

    return output


def _yaw_to_quaternion(yaw: float) -> Quaternion:
    """Convert yaw angle to quaternion using ROS2 tf_transformations.

    Args:
        yaw (float): Yaw angle in radians.

    Returns:
        Quaternion: Quaternion representing the yaw angle.
    """
    q = Quaternion()
    q.x, q.y, q.z, q.w = quaternion_from_euler(0, 0, yaw)
    return q


def _to_traj(
    info: OriginalInfo,
    pred_traj: NDArray,
) -> Trajectory:
    """Convert prediction of a single mode to Trajectory msg.

    Args:
        info (OriginalInfo): Object original info.
        pred_score (float): Predicted score.
        pred_traj (NDArray): Predicted waypoints in the shape of (T, 4).
    Returns:
        Trajectory: Instanced msg.
    """
    output = Trajectory()
    time_step = 0.1  # TODO(ktro2828): use specific value?
    for i, mode_point in enumerate(pred_traj):  # (x, y, vx, vy)
        x, y, _, _, _, vx, vy = mode_point
        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = info.kinematics.pose_with_covariance.pose.position.z
        if (i == 0):
            pose.orientation = info.kinematics.pose_with_covariance.pose.orientation
        else:
            prev_x = output.points[i-1].pose.position.x
            prev_y = output.points[i-1].pose.position.y
            yaw = np.arctan2(y-prev_y, x-prev_x)
            pose.orientation = _yaw_to_quaternion(yaw)

        trajectory_point = TrajectoryPoint()
        trajectory_point.pose = pose
        trajectory_point.longitudinal_velocity_mps = float(vx)
        trajectory_point.lateral_velocity_mps = float(vy)
        trajectory_point.acceleration_mps2 = 0.0
        trajectory_point.time_from_start = Duration(seconds=time_step*i).to_msg()
        output.points.append(trajectory_point)

    return output
