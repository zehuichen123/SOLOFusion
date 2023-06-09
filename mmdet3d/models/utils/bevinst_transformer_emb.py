import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (MultiScaleDeformableAttention,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
from .bevinst_transformer import BEVInstCrossAtten

@TRANSFORMER.register_module()
class BEVInstEmbTransformer(BaseModule):
    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 num_proposal=0,
                 two_stage_num_proposals=300,
                 decoder=None,
                 **kwargs):
        super(BEVInstEmbTransformer, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.num_proposal = num_proposal
        self.two_stage_num_proposals = two_stage_num_proposals
        # self.init_layers()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention) or isinstance(m, BEVInstCrossAtten):
                m.init_weight()
        # xavier_init(self.refpos_embedding, distribution='uniform', bias=0.)

    def forward(self,
                mlvl_feats,
                query_embed,
                reg_branches=None,
                priors=None,
                **kwargs):
        if query_embed is not None:
            query = query_embed['query_embed']
            bboxes = query_embed['bboxes']
            bs = mlvl_feats[0].size(0)
            query = query.unsqueeze(0).expand(bs, -1, -1)
            bboxes = bboxes.unsqueeze(0).expand(bs, -1, -1)
            reference_points = bboxes[..., [0,1,4]].sigmoid()
            # query_pos = self.refpos_embedding(reference_points)
            query_pos = self.decoder.box_encode(bboxes)
        
        if priors is not None:
            prev_query_feat = priors['query_embed']
            prev_bboxes = priors['bboxes']
            prev_reference_points = prev_bboxes[..., [0,1,4]].sigmoid()
            # prev_query_pos = self.refpos_embedding(prev_reference_points)
            prev_query_pos = self.decoder.box_encode(prev_bboxes)
            
            if query_embed is not None:
                query = torch.cat([prev_query_feat, query], dim=1)
                query_pos = torch.cat([prev_query_pos, query_pos], dim=1)
                bboxes = torch.cat([prev_bboxes, bboxes], dim=1)
                reference_points = torch.cat(
                    [prev_reference_points, reference_points], dim=1)
            else:
                query = prev_query_feat
                query_pos = prev_query_pos
                bboxes = prev_bboxes
                reference_points = prev_reference_points

        # decoder
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_bboxes = self.decoder(
            query=query,
            key=None,
            value=mlvl_feats,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            bboxes=bboxes,
            **kwargs)
        
        return inter_states, inter_bboxes

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVInstEmbTransformerDecoder(TransformerLayerSequence):
    def __init__(self, *args, transformerlayers_twin=None, return_intermediate=False,
                 multi_offset=False, normalize_yaw=False, **kwargs):
        super(BEVInstEmbTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.multi_offset = multi_offset
        self.normalize_yaw = normalize_yaw

        if transformerlayers_twin is not None:
            import copy
            from mmcv.cnn.bricks.transformer import build_transformer_layer
            from mmcv.runner.base_module import ModuleList
            if isinstance(transformerlayers_twin, dict):
                transformerlayers_twin = [
                    copy.deepcopy(transformerlayers_twin) for _ in range(self.num_layers)
                ]
            else:
                assert isinstance(transformerlayers_twin, list) and \
                    len(transformerlayers_twin) == self.num_layers
            self.layers_twin = ModuleList()
            for i in range(self.num_layers):
                self.layers_twin.append(build_transformer_layer(transformerlayers_twin[i]))
            self.num_layers = self.num_layers*2
        else:
            self.layers_twin = None
            
        def embedding_layer(input_dims):
            return  nn.Sequential(
                nn.Linear(input_dims, self.embed_dims),
                nn.ReLU(inplace=True),
                nn.LayerNorm(self.embed_dims),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(inplace=True),
                nn.LayerNorm(self.embed_dims),
            )
        self.pos_fc = embedding_layer(3)
        self.size_fc = embedding_layer(3)
        self.yaw_fc = embedding_layer(2)
        self.vel_fc = embedding_layer(2)
        self.output_fc = embedding_layer(self.embed_dims)
        self.init_weights()
        
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        xavier_init(self.pos_fc, distribution='uniform', bias=0.)
        xavier_init(self.size_fc, distribution='uniform', bias=0.)
        xavier_init(self.yaw_fc, distribution='uniform', bias=0.)
        xavier_init(self.vel_fc, distribution='uniform', bias=0.)
        xavier_init(self.output_fc, distribution='uniform', bias=0.)
        
    def box_encode(self, box_3d, X=0, Y=1, Z=4, W=2, L=3, H=5, SIN_Y=6, COS_Y=7, V=8):
        pos_feat = self.pos_fc(box_3d[..., [X, Y, Z]])
        size_feat = self.size_fc(box_3d[..., [W, L, H]])
        yaw_feat = self.yaw_fc(box_3d[..., [SIN_Y, COS_Y]])
        output = pos_feat + size_feat + yaw_feat
        vel_feat = self.vel_fc(box_3d[..., V:])
        output = output + vel_feat
        output = self.output_fc(output)
        return output

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                reference_points=None,
                reg_branches=None,
                bboxes=None,
                **kwargs):
        output = query
        velocity = bboxes[..., [8,9]]
        intermediate = []
        intermediate_bboxes = []
        for i, layer in enumerate(self.layers):
            lid = 2*i if self.layers_twin is not None else i
            reference_points_input = reference_points
            velocity_input = velocity
            output = layer(
                output,
                key=key,
                value=value,
                query_pos=query_pos,
                reference_points=reference_points_input,
                velocity=velocity_input,
                **kwargs)
            output = output.permute(1, 0, 2)
            query_pos = query_pos.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output+query_pos)
                # bboxes = 0:x, 1:y, 2:l, 3:w, 4:z, 5:h, 6:sin, 7:cos, 8:vx, 9:vy
                if self.multi_offset:
                    bboxes[..., :6] = bboxes[..., :6] + tmp[..., :6]
                    if self.normalize_yaw:
                        bboxes[..., 6:8] = F.normalize(tmp[..., 6:8], dim=-1)
                    else:
                        bboxes[..., 6:8] = tmp[..., 6:8]
                    bboxes[..., 8:] = tmp[..., 8:] / 0.5
                else:
                    bboxes = bboxes + tmp
                new_reference_points = bboxes[..., [0,1,4]].sigmoid()
                reference_points = new_reference_points.detach()
                velocity = bboxes[..., [8,9]].detach()
                query_pos = self.box_encode(bboxes)

            output = output.permute(1, 0, 2)
            query_pos = query_pos.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_bboxes.append(bboxes)

            # ---------------------------------------------------------
            if self.layers_twin is not None:
                reference_points_input = reference_points
                velocity_input = velocity
                output = self.layers_twin[i](
                    output,
                    key=key,
                    value=value,
                    query_pos=query_pos,
                    reference_points=reference_points_input,
                    velocity=velocity_input,
                    **kwargs)
                output = output.permute(1, 0, 2)
                query_pos = query_pos.permute(1, 0, 2)

                if reg_branches is not None:
                    tmp = reg_branches[2*i+1](output+query_pos)
                    # bboxes = 0:x, 1:y, 2:l, 3:w, 4:z, 5:h, 6:sin, 7:cos, 8:vx, 9:vy
                    if self.multi_offset:
                        bboxes[..., :6] = bboxes[..., :6] + tmp[..., :6]
                        if self.normalize_yaw:
                            bboxes[..., 6:8] = F.normalize(tmp[..., 6:8], dim=-1)
                        else:
                            bboxes[..., 6:8] = tmp[..., 6:8]
                        bboxes[..., 8:] = tmp[..., 8:] / 0.5
                    else:
                        bboxes = bboxes + tmp
                    new_reference_points = bboxes[..., [0,1,4]].sigmoid()
                    reference_points = new_reference_points.detach()
                    velocity = bboxes[..., [8,9]].detach()
                    query_pos = self.box_encode(bboxes)

                output = output.permute(1, 0, 2)
                query_pos = query_pos.permute(1, 0, 2)
                if self.return_intermediate:
                    intermediate.append(output)
                    intermediate_bboxes.append(bboxes)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_bboxes)

        return output, bboxes
