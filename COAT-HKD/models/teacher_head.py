# This file is part of COAT, and is distributed under the
# OSI-approved BSD 3-Clause License. See top-level LICENSE file or
# https://github.com/Kitware/COAT/blob/master/LICENSE for details.

from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn import init

from loss.oim import OIMLoss
from models.resnet import build_resnet
from models.transformer import TransformerHead


class COAT_REID(nn.Module):
    def __init__(self, cfg):
        super(COAT_REID, self).__init__()

        backbone, _ = build_resnet(name="resnet50", pretrained=True)
        box_head_3rd = TransformerHead(
            cfg=cfg,
            trans_names=cfg.MODEL.TRANSFORMER.NAMES_3RD,
            kernel_size=cfg.MODEL.TRANSFORMER.KERNEL_SIZE_3RD,
            use_feature_mask=cfg.MODEL.TRANSFORMER.USE_MASK_3RD,
        )

        roi_heads = CascadedROIHeads(cfg=cfg, box_head_3rd=box_head_3rd)

        self.backbone = backbone

        self.roi_heads = roi_heads
        self.eval_feat = cfg.EVAL_FEATURE
        self.lw_rcnn_reid_3rd = cfg.SOLVER.LW_RCNN_REID_3RD

    def forward(self, features, targets=None, query_img_as_gallery=False):
        loss_rcnn_reid_3rd, feats_reid_3rd, targets_reid_3rd, logits = self.roi_heads(features, targets)

        # apply loss weights
        # losses["loss_rcnn_reid_3rd"] *= self.lw_rcnn_reid_3rd

        return loss_rcnn_reid_3rd, feats_reid_3rd, targets_reid_3rd, logits


class CascadedROIHeads(nn.Module):
    '''
    https://github.com/pytorch/vision/blob/master/torchvision/models/detection/roi_heads.py
    '''

    def __init__(self, cfg, box_head_3rd):
        super(CascadedROIHeads, self).__init__()
        # ROI head
        self.use_diff_thresh = cfg.MODEL.ROI_HEAD.USE_DIFF_THRESH

        # Transformer head
        self.box_head_3rd = box_head_3rd

        # Feature embedding
        embedding_dim = cfg.MODEL.EMBEDDING_DIM
        self.embedding_head_2nd = NormAwareEmbedding(featmap_names=["before_trans", "after_trans"],
                                                     in_channels=[1024, 2048], dim=embedding_dim)
        self.embedding_head_3rd = deepcopy(self.embedding_head_2nd)
        self.pool = nn.AdaptiveAvgPool2d(14)
        # OIM
        num_pids = cfg.MODEL.LOSS.LUT_SIZE
        num_cq_size = cfg.MODEL.LOSS.CQ_SIZE
        oim_momentum = cfg.MODEL.LOSS.OIM_MOMENTUM
        oim_scalar = cfg.MODEL.LOSS.OIM_SCALAR
        self.reid_loss_3rd = OIMLoss(embedding_dim, num_pids, num_cq_size, oim_momentum, oim_scalar)

        # evaluation
        self.eval_feat = cfg.EVAL_FEATURE

    def forward(self, features, targets=None):
        """
        Arguments:
            features (List[Tensor])
            boxes (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """

        # --------------------- The third stage -------------------- #
        box_features = self.box_head_3rd(features)
        box_embeddings_3rd, box_cls_scores_3rd = self.embedding_head_3rd(box_features)

        loss_rcnn_reid_3rd, feats_reid_3rd, targets_reid_3rd, logits = self.reid_loss_3rd(box_embeddings_3rd, targets)
        return loss_rcnn_reid_3rd, feats_reid_3rd, targets_reid_3rd, logits


class NormAwareEmbedding(nn.Module):
    """
    Implements the Norm-Aware Embedding proposed in
    Chen, Di, et al. "Norm-aware embedding for efficient person search." CVPR 2020.
    """

    def __init__(self, featmap_names=["feat_res4", "feat_res5"], in_channels=[1024, 2048], dim=256):
        super(NormAwareEmbedding, self).__init__()
        self.featmap_names = featmap_names
        self.in_channels = in_channels
        self.dim = dim

        self.projectors = nn.ModuleDict()
        indv_dims = self._split_embedding_dim()
        for ftname, in_channel, indv_dim in zip(self.featmap_names, self.in_channels, indv_dims):
            proj = nn.Sequential(nn.Linear(in_channel, indv_dim), nn.BatchNorm1d(indv_dim))
            init.normal_(proj[0].weight, std=0.01)
            init.normal_(proj[1].weight, std=0.01)
            init.constant_(proj[0].bias, 0)
            init.constant_(proj[1].bias, 0)
            self.projectors[ftname] = proj

        self.rescaler = nn.BatchNorm1d(1, affine=True)

    def forward(self, featmaps):
        """
        Arguments:
            featmaps: OrderedDict[Tensor], and in featmap_names you can choose which
                      featmaps to use
        Returns:
            tensor of size (BatchSize, dim), L2 normalized embeddings.
            tensor of size (BatchSize, ) rescaled norm of embeddings, as class_logits.
        """
        assert len(featmaps) == len(self.featmap_names)
        if len(featmaps) == 1:
            k, v = featmaps.items()[0]
            v = self._flatten_fc_input(v)
            embeddings = self.projectors[k](v)
            norms = embeddings.norm(2, 1, keepdim=True)
            embeddings = embeddings / norms.expand_as(embeddings).clamp(min=1e-12)
            norms = self.rescaler(norms).squeeze()
            return embeddings, norms
        else:
            outputs = []
            for k, v in featmaps.items():
                v = self._flatten_fc_input(v)
                outputs.append(self.projectors[k](v))
            embeddings = torch.cat(outputs, dim=1)
            norms = embeddings.norm(2, 1, keepdim=True)
            embeddings = embeddings / norms.expand_as(embeddings).clamp(min=1e-12)
            norms = self.rescaler(norms).squeeze()
            return embeddings, norms

    def _flatten_fc_input(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            return x.flatten(start_dim=1)
        return x

    def _split_embedding_dim(self):
        parts = len(self.in_channels)
        tmp = [self.dim // parts] * parts
        if sum(tmp) == self.dim:
            return tmp
        else:
            res = self.dim % parts
            for i in range(1, res + 1):
                tmp[-i] += 1
            assert sum(tmp) == self.dim
            return tmp
