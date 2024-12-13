import os.path as osp
import hashlib

from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
import rclpy.duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
import rclpy.parameter
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import yaml
import torch


from awml_pred.common import Config, create_logger, get_num_devices, init_dist_pytorch, init_dist_slurm, load_checkpoint
from awml_pred.models import build_model
from awml_pred.deploy.apis.torch2onnx import _load_inputs
from autoware_mtr.conversion.ego import from_odometry
from autoware_mtr.conversion.misc import timestamp2ms
from autoware_mtr.datatype import AgentLabel
from autoware_mtr.dataclass.history import AgentHistory

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

        self._history = AgentHistory(max_length=num_timestamp)


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

        if build_only:
            exit(0)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

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
        # print(self.model(**dummy_input))


        # pre-process
        # inputs = self._preprocess(self._history, current_ego, self._lane_segments)

        # # inference

        with torch.no_grad():
            pred_scores, pred_trajs = self.model(**dummy_input)

        print("pred_scores.shape() ",pred_scores.shape)
        print("pred_trajs.shape()", pred_trajs.shape)

        # # post-process
        # pred_scores, pred_trajs = self._postprocess(pred_scores, pred_trajs)

        # # convert to ROS msg
        # pred_objs = to_predicted_objects(
        #     header=msg.header,
        #     infos=[info],
        #     pred_scores=pred_scores,
        #     pred_trajs=pred_trajs,
        #     score_threshold=self._score_threshold,
        # )
        # self._publisher.publish(pred_objs)

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
        assert num_feat == 7, f"Expected predicted feature is (x, y, vx, vy), but got {num_feat}"

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
