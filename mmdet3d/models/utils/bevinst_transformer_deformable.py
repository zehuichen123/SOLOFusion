import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.runner.base_module import BaseModule
from .bevinst_transformer import FeedForward


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@ATTENTION.register_module()
class BEVInstCrossAtten_deform(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=5,
                 num_cams=6,
                 num_frames=9,
                 temporal_weight=0.9 ** (0.5 * 10),
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None,
                 batch_first=False):
        super(BEVInstCrossAtten_deform, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            import warnings
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams
        self.num_frames = num_frames
        self.attention_weights = nn.Linear(embed_dims,
                                           num_cams * num_frames * num_levels * num_points)
        if self.num_frames > 1:
            self.temporal_weight = temporal_weight
            self.temporal_net = FeedForward(embed_dims * 2, embed_dims * 4, embed_dims)
        if num_points > 1:
            self.sampling_offsets = nn.Linear(embed_dims, num_cams * num_frames * num_levels * (num_points - 1) * 2)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.batch_first = batch_first

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                velocity=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)

        bs, num_query, _ = query.size()

        attention_weights = self.attention_weights(query).view(bs, 1, num_query, self.num_cams, self.num_frames, self.num_points,
                                                               self.num_levels)
        # note: deformable
        if self.num_points > 1:
            sampling_offsets = self.sampling_offsets(query).view(bs * self.num_cams * self.num_frames, num_query, (self.num_points - 1), 2,
                                                                 self.num_levels)
        else:
            sampling_offsets = None

        reference_points_3d, output, mask = feature_sampling_4d(value, reference_points, velocity, self.pc_range, kwargs['img_metas'],
                                                                sampling_offsets, self.num_points)
        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)

        attention_weights = attention_weights.sigmoid() * mask
        output = output * attention_weights
        output = output.sum(-1).sum(-3)  # scale view
        output = output.permute(3, 4, 2, 0, 1)
        # B C N T K -> T K N B C -> K N B C
        output_t = output[0]
        if self.num_frames > 1:
            for i in range(1, self.num_frames):
                output_aux = output[i] * self.temporal_weight
                output_t = torch.cat([output_t, output_aux], -1)
                output_t = self.temporal_net(output_t)
        output = output_t.sum(0)  # keypoint

        output = self.output_proj(output)
        # (num_query, bs, embed_dims)
        pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)

        return self.dropout(output) + inp_residual + pos_feat


def feature_sampling_4d(mlvl_feats, reference_points, velocity, pc_range, img_metas, sampling_offsets, num_points):
    rt2img = torch.stack([img_meta['rt2img'] for img_meta in img_metas])
    num_frame = rt2img.shape[2]
    rt2img = reference_points.new_tensor(rt2img)  # (B, N, T, 4, 4)
    reference_points = reference_points.clone()
    reference_points_3d = reference_points.clone()
    velocity = velocity.clone()
    reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
    B, num_query = reference_points.size()[:2]

    reference_points = reference_points.unsqueeze(2).repeat(1, 1, num_frame, 1)
    for t in range(num_frame):
        reference_points[..., t, :2] = reference_points[..., t, :2] - 0.5 * t * velocity
    reference_points = reference_points.permute(0, 2, 1, 3)
    if 'bda' in img_metas[0].keys():
        bda_inv = torch.stack([img_meta['bda'].inverse() for img_meta in img_metas])
        bda_inv = bda_inv.view(B, 1, 1, 3, 3).repeat(1, num_frame, num_query, 1, 1)
        reference_points = torch.matmul(bda_inv, reference_points.unsqueeze(-1)).squeeze(-1)
    # reference_points (B, num_frame, num_queries, 4)
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
    num_cam = rt2img.size(1)
    reference_points = reference_points[:, None].repeat(1, num_cam, 1, 1, 1).unsqueeze(-2)
    rt2img = rt2img.view(B, num_cam, num_frame, 1, 4, 4).repeat(1, 1, 1, num_query, 1, 1)
    reference_points_cam = torch.matmul(reference_points, rt2img).squeeze(-2)
    eps = 1e-5
    # mask = (reference_points_cam[..., 2:3] > eps)
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)
    post_rot_xy_T = torch.stack([img_meta['post_rot_xy_T'] for img_meta in img_metas])
    post_tran_xy = torch.stack([img_meta['post_tran_xy'] for img_meta in img_metas])
    post_rot_xy_T = post_rot_xy_T.view(B, num_cam, num_frame, 1, 2, 2).repeat(1, 1, 1, num_query, 1, 1)
    post_tran_xy = post_tran_xy.view(B, num_cam, num_frame, 1, 2)
    reference_points_cam = torch.matmul(reference_points_cam.unsqueeze(-2),
                                        post_rot_xy_T).squeeze(-2) + post_tran_xy

    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][1]
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0]
    reference_points_cam = (reference_points_cam - 0.5) * 2
    # mask = (mask & (reference_points_cam[..., 0:1] > -1.0)
    #         & (reference_points_cam[..., 0:1] < 1.0)
    #         & (reference_points_cam[..., 1:2] > -1.0)
    #         & (reference_points_cam[..., 1:2] < 1.0))
    # mask = mask.view(B, num_cam, num_frame, 1, num_query, 1, 1).permute(0, 3, 4, 1, 2, 5, 6)  # B 1 900 6 9 1 1
    # mask = torch.nan_to_num(mask)
    sampled_feats = []
    mask_deformable = []
    for lvl, feat in enumerate(mlvl_feats):
        B, N, T, C, H, W = feat.size()
        feat = feat.contiguous().view(B * N * T, C, H, W)
        reference_points_cam_lvl = reference_points_cam.view(B * N * T, num_query, 1, 2)
        center = reference_points_cam_lvl.clone()
        if sampling_offsets is not None:
            reference_points_cam_lvl = reference_points_cam_lvl + sampling_offsets[..., lvl]
            reference_points_cam_lvl = torch.cat((reference_points_cam_lvl, center), dim=-2)
        mask_deformable.append(((reference_points_cam_lvl < 1) * (reference_points_cam_lvl > -1)).all(-1).unsqueeze(-1))
        sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
        sampled_feat = sampled_feat.view(B, N, T, C, num_query, num_points).permute(0, 3, 4, 1, 2, 5)
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1)
    # bs, 1, num_query, self.num_cams, self.num_frames, self.num_points, self.num_levels
    mask_deformable = torch.cat(mask_deformable, dim=-1).view(B, num_cam, 1, num_query,num_frame, num_points, len(mlvl_feats)).permute(0, 2, 3, 1, 4, 5, 6)
    # 1 为sample的组数  即keypoint
    sampled_feats = sampled_feats.view(B, C, num_query, num_cam, num_frame, num_points, len(mlvl_feats))
    return reference_points_3d, sampled_feats, mask_deformable
