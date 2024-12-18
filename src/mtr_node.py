import os.path as osp
import hashlib
import yaml
import torch
import numpy as np
from collections import deque

import rclpy
import rclpy.duration
import rclpy.parameter
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from nav_msgs.msg import Odometry

from numpy.typing import NDArray
from rcl_interfaces.msg import ParameterDescriptor

from autoware_perception_msgs.msg import PredictedObjects
from awml_pred.dataclass import Trajectory
from awml_pred.common import Config, create_logger, get_num_devices, init_dist_pytorch, init_dist_slurm, load_checkpoint
from awml_pred.models import build_model
from awml_pred.deploy.apis.torch2onnx import  _load_inputs
from autoware_mtr.conversion.ego import from_odometry
from autoware_mtr.conversion.misc import timestamp2ms
from autoware_mtr.conversion.lanelet import convert_lanelet
from autoware_mtr.conversion.trajectory import get_relative_history
from autoware_mtr.datatype import AgentLabel
from autoware_mtr.geometry import rotate_along_z
from autoware_mtr.dataclass.history import AgentHistory
from autoware_mtr.dataclass.lane import LaneSegment
from autoware_mtr.dataclass.agent import AgentState
from autoware_mtr.preprocess import embed_agent , embed_polyline , relative_pose_encode
from autoware_mtr.conversion.predicted_object import to_predicted_objects
from typing_extensions import Self
from dataclasses import dataclass

@dataclass
class ModelInput:
    uuids: list[str]
    actor: NDArray
    lane: NDArray
    rpe: NDArray
    rpe_mask: NDArray | None = None

    def cuda(self, device: int | torch.device | None = None) -> Self:
        self.actor = torch.from_numpy(self.actor).cuda(device)
        self.lane = torch.from_numpy(self.lane).cuda(device)
        self.rpe = torch.from_numpy(self.rpe).cuda(device)
        if self.rpe_mask is not None:
            self.rpe_mask = torch.from_numpy(self.rpe_mask).cuda(device)

        return self


def softmax(x: NDArray, axis: int) -> NDArray:
    """Apply softmax.

    Args:
        x (NDArray): Input array.
        axis (int): Axis to apply softmax.

    Returns:
        NDArray: Softmax result.
    """
    x -= x.max(axis=axis, keepdims=True)
    x_exp = np.exp(x)
    return x_exp / x_exp.sum(axis=axis, keepdims=True)

class MTRNode(Node):
    def __init__(self) -> None:
        super().__init__("mtr_python_node")

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        #subscribers
        self._subscription = self.create_subscription(
            Odometry,
            "~/input/ego",
            self._callback,
            qos_profile,
        )

        # ROS parameters
        descriptor = ParameterDescriptor(dynamic_typing=True)

        build_only = (
            self.declare_parameter("build_only", descriptor=descriptor)
            .get_parameter_value()
            .bool_value
        )

        model_config_path = (
            self.declare_parameter("model_config", descriptor=descriptor)
            .get_parameter_value()
            .string_value
        )

        deploy_config_path = (
            self.declare_parameter("deploy_config", descriptor=descriptor)
            .get_parameter_value()
            .string_value
        )

        checkpoint_path = (
            self.declare_parameter("checkpoint_path", descriptor=descriptor)
            .get_parameter_value()
            .string_value
        )

        num_timestamp = (
            self.declare_parameter("num_timestamp", descriptor=descriptor)
            .get_parameter_value()
            .integer_value
        )

        self._timestamp_threshold = (
            self.declare_parameter("timestamp_threshold", descriptor=descriptor)
            .get_parameter_value()
            .double_value
        )

        self._score_threshold = (
            self.declare_parameter("score_threshold", descriptor=descriptor)
            .get_parameter_value()
            .double_value
        )

        lanelet_file = (
            self.declare_parameter("lanelet_file", descriptor=descriptor)
            .get_parameter_value()
            .string_value
        )

        labels = (
            self.declare_parameter("labels", descriptor=descriptor)
            .get_parameter_value()
            .string_array_value
        )

        self._history = AgentHistory(max_length=num_timestamp)

        self._lane_segments: list[LaneSegment] = convert_lanelet(lanelet_file)

        self._label_ids = [AgentLabel.from_str(label).value for label in labels]

        cfg = Config.from_file(model_config_path)
        is_distributed = True

        #Ego info
        self._ego_uuid = hashlib.shake_256("EGO".encode()).hexdigest(8)


        # Load Model
        self.model = build_model(cfg.model)
        self.model.eval()
        self.model.cuda()
        self.model, _ = load_checkpoint(self.model, checkpoint_path, is_distributed=is_distributed)
        self.deploy_cfg = Config.from_file(deploy_config_path)
        self.count = 0
        if build_only:
            exit(0)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        # publisher
        self._publisher = self.create_publisher(PredictedObjects, "~/output/objects", qos_profile)

    def _callback(self, msg: Odometry) -> None:
        # remove invalid ancient agent history
        timestamp = timestamp2ms(msg.header)
        self._history.remove_invalid(timestamp, self._timestamp_threshold)

        # update agent history
        current_ego, info = from_odometry(
            msg,
            uuid=self._ego_uuid,
            label_id=AgentLabel.VEHICLE,
            size=(4.0, 2.0, 1.0),  # size is unused dummy
        )
        # print("current_ego.xyz",current_ego.xyz)
        self._history.update_state(current_ego, info)

        dummy_input = _load_inputs(self.deploy_cfg.input_shapes)

        # pre-process
        past_embed = self._preprocess(self._history, current_ego, self._lane_segments)
        if self.count > 11:
            dummy_input["obj_trajs"] = torch.Tensor(past_embed).cuda()
            print(" dummy_input[obj_trajs]",  dummy_input["obj_trajs"].shape)

        if self.count <= 11:
            self.count = self.count + 1
        # # inference

        with torch.no_grad():
            pred_scores, pred_trajs = self.model(**dummy_input)

        # print("pred_scores.shape() ", pred_scores.shape)
        # print("pred_trajs.shape()", pred_trajs.shape)

        # print("---------dummy----------")
        # for key, value in dummy_input.items():
        #     print(key, value.shape)
        # print("----------dummy----------")

        # print("inputs.actor", inputs.actor.shape)
        # print("inputs.lane", inputs.lane.shape)
        # print("inputs.rpe", inputs.rpe.shape)
        # print("inputs.rpe_mask", inputs.rpe_mask.shape)


        # # post-process
        pred_scores, pred_trajs = self._postprocess(pred_scores, pred_trajs)

        # # convert to ROS msg
        pred_objs = to_predicted_objects(
            header=msg.header,
            infos=[info],
            pred_scores=pred_scores,
            pred_trajs=pred_trajs,
            score_threshold=self._score_threshold,
        )
        self._publisher.publish(pred_objs)

    def _postprocess(
        self,
        pred_scores: NDArray | torch.Tensor,
        pred_trajs: NDArray | torch.Tensor,
    ) -> tuple[NDArray, NDArray]:
        """Run postprocess.

        Args:
            pred_scores (NDArray | torch.Tensor): Predicted scores in the shape of
                (N, M).
            pred_trajs (NDArray | torch.Tensor): Predicted trajectories in the shape of
                (N, M, T, 4).

        Returns:
            tuple[NDArray, NDArray]: Transformed and sorted prediction.
        """
        if isinstance(pred_scores, torch.Tensor):
            pred_scores = pred_scores.cpu().detach().numpy()
        if isinstance(pred_trajs, torch.Tensor):
            pred_trajs = pred_trajs.cpu().detach().numpy()
        # predicted traj point info is X,Y,Xmean,Ymean,Variance,Vx,Vy

        num_agent, num_mode, num_future, num_feat = pred_trajs.shape
        assert num_feat == 7, f"Expected predicted feature is (X, Y, Xmean, Ymean, Variance, Vx, Vy), but got {num_feat}"

        # transform from agent centric coords to world coords
        current_agent, _ = self._history.as_trajectory(latest=True)

        #first 2 elements are xy then we use the negative ego rotation. reshape, I dont know
        # each trajectory is in  the ref frame of its own agent. For ego we might use TF ?
        pred_trajs[..., :2] = rotate_along_z(
            pred_trajs.reshape(num_agent, -1, num_feat)[..., :2], -current_agent.yaw
        ).reshape(num_agent, num_mode, num_future, 2)
        pred_trajs[..., :2] += current_agent.xy[:, None, None, :]

        # sort by score
        pred_scores = softmax(pred_scores, axis=1)
        sort_indices = np.argsort(-pred_scores, axis=1)
        pred_scores = np.take_along_axis(pred_scores, sort_indices, axis=1)
        pred_trajs = np.take_along_axis(pred_trajs, sort_indices[..., None, None], axis=1)

        return pred_scores, pred_trajs


    def get_ego_past(self, ego_history :  deque[AgentState]):
        """
        each state contains
        uuid: str
        timestamp: float = -np.inf
        label_id: int = -1
        xyz: NDArray = np.zeros(3)
        size: NDArray = np.zeros(3)
        yaw: float = 0.0
        vxy: NDArray = np.zeros(2)
        is_valid: bool = False"""

        num_target, num_agent, num_time = 1 , 1 , 11
        num_type = 3

        ego_past_xyz = np.ones((num_target, num_agent, num_time, 3), dtype=np.float32)
        ego_past_Vxy = np.ones((num_target, num_agent, num_time, 3), dtype=np.float32)
        ego_past_xyz_size = np.ones((num_target, num_agent, num_time, 1), dtype=np.int32)

        yaw_embed = np.ones((num_target, num_agent, num_time, 2), dtype=np.float32)

        # accel
        # TODO: use accurate timestamp diff
        # vel_diff = np.diff(agent_past.vxy, axis=2, prepend=agent_past.vxy[..., 0, :][:, :, None, :])
        # accel = vel_diff / 0.1
        # accel[:, :, 0, :] = accel[:, :, 1, :]
        ego_timestamps = np.ones((num_time), dtype=np.float32)
        for i,ego_state in enumerate(ego_history):
            # ego_timestamps[i] = ego_state.timestamp
            ego_past_xyz[0,0,i,0] = ego_state.xyz[0]
            ego_past_xyz[0,0,i,1] = ego_state.xyz[1]
            ego_past_xyz[0,0,i,2] = ego_state.xyz[2]

            yaw_embed[0,0, i,0] = np.sin(ego_state.yaw)
            yaw_embed[0,0,i, 1] = np.cos(ego_state.yaw)

            ego_past_Vxy[0,0,i,0] = ego_state.vxy[0]
            ego_past_Vxy[0,0,i,1] = ego_state.vxy[1]
            ego_timestamps[i] = ego_state.timestamp

        time_embed = np.zeros((num_target, num_agent, num_time, num_time + 1), dtype=np.float32)
        time_embed[:, :, np.arange(num_time), np.arange(num_time)] = 1
        time_embed[0, 0, :num_time, -1] = ego_timestamps

        types = ["TYPE_VEHICLE", "TYPE_PEDESTRIAN", "TYPE_CYCLIST"]
        type_onehot = np.zeros((num_target, num_agent, num_time, num_type + 2), dtype=np.float32)
        for i, target_type in enumerate(types):
            type_onehot[:, "TYPE_VEHICLE" == target_type, :, i] = 1
        type_onehot[np.arange(num_target), 0, :, num_type] = 1 ## target indices replaced by 0
        type_onehot[:,0 , :, num_type + 1] = 1             # scenario.ego_index replaced by 0
        accel = np.zeros((num_target, num_agent, num_time, 3), dtype=np.float32)
        # vel_diff = np.diff(ego_past_Vxy, axis=2, prepend=ego_past_Vxy[..., 0, :][:, :, None, :])
        # accel = vel_diff / 0.1
        # accel[:, :, 0, :] = accel[:, :, 1, :]

        past_embed = np.concatenate(
            (
                ego_past_xyz,
                ego_past_xyz_size,
                type_onehot,
                time_embed,
                yaw_embed,
                ego_past_Vxy,
                accel,
            ),
            axis=-1,
            dtype=np.float32,
        )
        # print("past embed \n",past_embed)
        return past_embed



    def _preprocess(
        self,
        history: AgentHistory,
        current_ego: AgentState,
        lane_segments: list[LaneSegment],
    ) -> ModelInput:
        """Run preprocess.

        Args:
            history (AgentHistory): Ego history.
            current_ego (AgentState): Current ego state.
            lane_segments (list[LaneSegments]): Lane segments.

        Returns:
            ModelInput: Model inputs.
        """
        # def get_current_ego_input(current_ego_state):
        #     agent_current_xyz = np.zeros((1, 1, 3), dtype=np.float32)
        #     agent_current_xyz[...,0] = current_ego_state.xy[0]
        #     agent_current_xyz[...,1] = current_ego_state.xy[1]
        #     agent_current_xyz[...,2] = 0.0
        #     return agent_current_xyz

        # ego_input = get_current_ego_input(current_ego)
        relative_history = get_relative_history(current_ego,self._history.histories[self._ego_uuid])
        past_embed = self.get_ego_past(relative_history)

        # print("past_embed", past_embed)
        # print("ego_input", ego_input)
        # print("past_embed shape", past_embed.shape)
        # print("ego_input shape ", ego_input.shape)
        return past_embed


def main(args=None) -> None:
    rclpy.init(args=args)

    node = MTRNode()
    executor = MultiThreadedExecutor()
    try:
        rclpy.spin(node, executor)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
