import torch
from torch import nn
from torch.nn.init import xavier_uniform_, constant_, normal_

from mmdet3d.models.builder import FUSERS

from ..utils.metafuser import SinePositionEmbedding, NaiveFuser

from ..ops.modules import MSDeformAttn

__all__ = ["MetaFuser"]


class BEVEvolvingBlock(nn.Sequential):
    def __init__(
        self,
        is_cross_attn=True,
        d_model=256,
        ffn_dim=256,
        n_heads=8,
        n_points=8,
        dropout=0.1,
    ):
        super().__init__()
        self.is_cross_attn = is_cross_attn

        self.norm1 = nn.LayerNorm(d_model)

        if self.is_cross_attn:
            self.cross_attn = MSDeformAttn(
                d_model, 1, n_heads, n_points
            )  # 1 is the number of levels
        else:
            # use deformable attn
            self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        self.norm2 = nn.LayerNorm(d_model)

        # TODO: choose a proper FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.ReLU(),
        )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def forward(self, tgt, query_pos, pos=None, src=None, reference_points=None, src_spatial_shape=None):
        """
        pos' use?

        tgt.shape: []
        query_pos.shape: []
        src.shape: []

        """
        
        tgt2 = self.norm1(tgt)

        if self.is_cross_attn:
            # reference_points = self.reference_points(src_spatial_shape, src.device)
            tgt2 = self.cross_attn(
                tgt2 + query_pos, reference_points, src, 
                torch.tensor([src_spatial_shape]), [0]
            )
        else:
            q = k = tgt2 + query_pos
            tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt2.transpose(0, 1))[0].transpose(0, 1)

        tgt = tgt + tgt2

        tgt2 = self.norm2(tgt)
        tgt2 = self.ffn(tgt2)

        return tgt + tgt2


@FUSERS.register_module()
class MetaFuser(nn.Module):
    def __init__(
        self,
        num_queries=180 * 180,
        in_channels=[80, 256],
        out_channels=256,
        d_model=256,
        nhead=8,
        dropout=0.1,
    ) -> None:
        super().__init__()
        """
        Mimicing Deformable DETR's decoder
        """

        # self.sinEmbed = SinePositionEmbedding(d_model // 2, normalize=True)
        self.out_channels = out_channels
        self.reference_points = nn.Linear(d_model, 2)
        self.query_embed = nn.Embedding(num_queries, 2 * d_model)

        self.cross_1 = BEVEvolvingBlock(is_cross_attn=True)
        self.cross_2 = BEVEvolvingBlock(is_cross_attn=True)
        self.cross_3 = BEVEvolvingBlock(is_cross_attn=True)

        self.fuser = NaiveFuser(in_channels, out_channels)

        self.self_1 = BEVEvolvingBlock(is_cross_attn=False)
        self.self_2 = BEVEvolvingBlock(is_cross_attn=False)

    def _reset_parameters(self):
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)

    def forward(self, features) -> torch.Tensor:
        cam_ft, lidar_ft = features
        # shape from (B, C, H, W) to (B, H*W, C)
        bs, c, h, w = cam_ft.shape
        src_spatial_shape = [h, w]
        flat_cam_ft = cam_ft.flatten(2).transpose(1, 2)
        flat_lidar_ft = lidar_ft.flatten(2).transpose(1, 2)

        # deformable detr doesnt seem to be using pos_embed
        # pos_embed = self.sinEmbed(torch.zeros(bs, self.out_channels, h, w))
        # pos_embed = pos_embed.to(flat_cam_ft.device) # BUG: may fuckup 
        # pos_embed = pos_embed.unsqueeze(0).expand(bs, -1, -1)

        query_embed = self.query_embed.weight
        query_embed, tgt = torch.split(query_embed, self.out_channels, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)

        reference_points = self.reference_points(query_embed).sigmoid()

        tgt = self.cross_1(
            tgt,
            src=flat_lidar_ft,
            # pos=pos_embed,
            query_pos=query_embed,
            reference_points=reference_points,
            src_spatial_shape=src_spatial_shape,
        )
        tgt = self.cross_2(
            tgt,
            src=flat_cam_ft,
            # pos=pos_embed,
            query_pos=query_embed,
            reference_points=reference_points,
            src_spatial_shape=src_spatial_shape,
        )
        tgt = self.cross_3(
            tgt,
            src=self.fuser([cam_ft, lidar_ft]),
            # pos=pos_embed,
            query_pos=query_embed,
            reference_points=reference_points,
            src_spatial_shape=src_spatial_shape,
        )

        tgt = self.self_1(tgt, pos=pos_embed)
        tgt = self.self_2(tgt, pos=pos_embed)

        return tgt
