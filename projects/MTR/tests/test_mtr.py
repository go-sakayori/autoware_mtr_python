import torch

from awml_pred.common import Config
from awml_pred.models import MTR
from awml_pred.test_utils import require_cuda


@require_cuda
def test_mtr() -> None:
    num_center_objects = 10
    num_objects = 15
    num_past_timestep = 11
    num_agent_dims = 29
    num_polylines = 768
    num_waypoints = 20
    num_polyline_dims = 9
    num_feature = 256
    num_future_frames = 80
    num_motion_modes = 6

    cfg = Config(
        dict(
            encoder=dict(
                name="MTREncoder",
                agent_polyline_encoder=dict(
                    name="PointNetPolylineEncoder",
                    in_channels=num_agent_dims + 1,
                    hidden_dim=256,
                    num_layers=3,
                    out_channels=num_feature,
                ),
                map_polyline_encoder=dict(
                    name="PointNetPolylineEncoder",
                    in_channels=9,
                    hidden_dim=64,
                    num_layers=5,
                    num_pre_layers=3,
                    out_channels=num_feature,
                ),
                attention_layer=dict(
                    name="TransformerEncoderLayer",
                    d_model=256,
                    num_head=8,
                    dim_feedforward=1024,
                    dropout=0.1,
                    normalize_before=False,
                    num_layers=6,
                ),
                use_local_attn=True,
                num_attn_neighbors=16,
            ),
            decoder=dict(
                name="MTRDecoder",
                in_channels=num_feature,
                num_future_frames=num_future_frames,
                num_motion_modes=num_motion_modes,
                d_model=512,
                num_decoder_layers=6,
                num_attn_head=8,
                map_center_offset=(30, 0),
                num_waypoint_map_polylines=128,
                num_base_map_polylines=256,
                dropout=0.1,
                map_d_model=256,
                nms_threshold=2.5,
                decode_loss=dict(
                    name="MTRLoss",
                    reg_cfg=dict(name="GMMLoss", weight=1.0, use_square_gmm=False),
                    cls_cfg=dict(name="CrossEntropyLoss", weight=1.0),
                    vel_cfg=dict(name="L1Loss", weight=0.2),
                ),
            ),
        ),
    )

    mtr = MTR(encoder=cfg.encoder, decoder=cfg.decoder)

    mtr = mtr.eval().cuda()

    # for encoder
    obj_trajs = torch.ones(num_center_objects, num_objects, num_past_timestep, num_agent_dims).cuda()
    obj_trajs_mask = torch.ones(num_center_objects, num_objects, num_past_timestep, dtype=torch.bool).cuda()
    map_polylines = torch.ones(num_center_objects, num_polylines, num_waypoints, num_polyline_dims).cuda()
    map_polylines_mask = torch.ones(num_center_objects, num_polylines, num_waypoints, dtype=torch.bool).cuda()
    map_polylines_center = torch.ones(num_center_objects, num_polylines, 3).cuda()
    obj_trajs_pos = torch.ones(num_center_objects, num_objects, 3).cuda()
    track_index_to_predict = torch.randint(0, 3, (num_center_objects,)).cuda()
    # for decoder
    intention_points = torch.rand(num_center_objects, 64, 2, dtype=torch.float32).cuda()

    scores, trajs = mtr(
        obj_trajs,
        obj_trajs_mask,
        map_polylines,
        map_polylines_mask,
        map_polylines_center,
        obj_trajs_pos,
        track_index_to_predict,
        intention_points,
    )

    assert scores.shape == (num_center_objects, num_motion_modes)
    assert trajs.shape == (num_center_objects, num_motion_modes, num_future_frames, 7)
