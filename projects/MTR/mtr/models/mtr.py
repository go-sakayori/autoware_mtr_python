from awml_pred.common import DETECTORS
from awml_pred.models.detectors import BaseDetector
from awml_pred.typing import Tensor

__all__ = ("MTR",)


@DETECTORS.register()
class MTR(BaseDetector):
    def __init__(self, encoder: dict, decoder: dict) -> None:
        super().__init__(encoder, decoder)

    def extract_feature(
        self,
        obj_trajs: Tensor,
        obj_trajs_mask: Tensor,
        map_polylines: Tensor,
        map_polylines_mask: Tensor,
        map_polylines_center: Tensor,
        obj_trajs_last_pos: Tensor,
        track_index_to_predict: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.encoder(
            obj_trajs,
            obj_trajs_mask,
            map_polylines,
            map_polylines_mask,
            map_polylines_center,
            obj_trajs_last_pos,
            track_index_to_predict,
        )

    def forward(
        self,
        obj_trajs: Tensor,
        obj_trajs_mask: Tensor,
        map_polylines: Tensor,
        map_polylines_mask: Tensor,
        map_polylines_center: Tensor,
        obj_trajs_last_pos: Tensor,
        track_index_to_predict: Tensor,
        intention_points: Tensor,
        center_gt_trajs: Tensor | None = None,
        center_gt_trajs_mask: Tensor | None = None,
        center_gt_final_valid_idx: Tensor | None = None,
        obj_trajs_future_state: Tensor | None = None,
        obj_trajs_future_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor] | dict:
        """Forward operation.

        Args:
        ----
            N: Number of center objects
            A: Number of objects
            T: Number of time steps
            Da: Number of objects dimensions
            K: Number of polylines
            P: Number of points
            Dp: Number of points dimensions

            * For encoder:
                * obj_trajs (Tensor): (N, A, T, Da)
                * obj_trajs_mask (Tensor): (N, A, T)
                * map_polylines (Tensor): (N, K, P, Dp)
                * map_polyline_mask (Tensor): (N, K, P)
                * obj_trajs_last_pos (Tensor): (N, A, 3)
                * track_index_to_predict (Tensor): (N,)

            * For decoder:
                * Encoder Output Feature
                * center_objects_type_idx (Tensor): (N,)

            * Only for training
                * center_gt_trajs (Tensor | None)
                * center_gt_trajs_mask (Tensor | None)
                * center_gt_final_valid_idx (Tensor | None)
                * obj_trajs_future_state (Tensor | None)
                * obj_trajs_future_mask (Tensor | None)

        Returns:
        -------
            tuple[Tensor, Tensor] | dict: In training, returns loss dict.
                Otherwise returns tensor of scores and trajectories.

        """
        obj_feature, obj_mask, obj_pos, map_feature, map_mask, map_pos, center_objects_feature = self.extract_feature(
            obj_trajs,
            obj_trajs_mask,
            map_polylines,
            map_polylines_mask,
            map_polylines_center,
            obj_trajs_last_pos,
            track_index_to_predict,
        )

        if self.training:
            loss: dict = self.decoder(
                obj_feature,
                obj_mask,
                obj_pos,
                map_feature,
                map_mask,
                map_pos,
                center_objects_feature,
                intention_points,
                center_gt_trajs,
                center_gt_trajs_mask,
                center_gt_final_valid_idx,
                obj_trajs_future_state,
                obj_trajs_future_mask,
            )
            return loss
        else:
            pred_scores, pred_trajs = self.decoder(
                obj_feature,
                obj_mask,
                obj_pos,
                map_feature,
                map_mask,
                map_pos,
                center_objects_feature,
                intention_points,
            )
            return pred_scores, pred_trajs
