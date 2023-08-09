from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from models.oim import OIMLoss
from models.resnet import build_resnet


class Seq_Reid(nn.Module):
    def __init__(self, cfg):
        super(Seq_Reid, self).__init__()

        backbone, reid_head, deep_reid_head = build_resnet(name="resnet50", pretrained=True)
        if cfg.use_deep_reid_head:
            reid_head=deep_reid_head

        roi_heads = SeqRoIHeads(
            # OIM
            num_pids=cfg.MODEL.LOSS.LUT_SIZE,
            num_cq_size=cfg.MODEL.LOSS.CQ_SIZE,
            oim_momentum=cfg.MODEL.LOSS.OIM_MOMENTUM,
            oim_scalar=cfg.MODEL.LOSS.OIM_SCALAR,
            reid_head=reid_head
        )

        self.roi_heads = roi_heads
        self.lw_box_reid = cfg.SOLVER.LW_BOX_REID

    def forward(self, features, targets=None):
        loss_box_reid, pid_inputs, pid_label, pid_logits = self.roi_heads(features, targets)
        return loss_box_reid, pid_inputs, pid_label, pid_logits


class SeqRoIHeads(nn.Module):
    def __init__(
            self,
            num_pids,
            num_cq_size,
            oim_momentum,
            oim_scalar,
            reid_head,
    ):
        super(SeqRoIHeads, self).__init__()
        self.embedding_head = NormAwareEmbedding()
        self.reid_loss = OIMLoss(256, num_pids, num_cq_size, oim_momentum, oim_scalar)
        self.reid_head = reid_head

    def forward(self, features, targets=None):
        # --------------------- Baseline head -------------------- #
        box_features = features
        box_features = self.reid_head(box_features)
        box_embeddings, box_cls_scores = self.embedding_head(box_features)

        losses = {}
        loss_box_reid, pid_inputs, pid_label, pid_logits = self.reid_loss(box_embeddings, targets)
        return loss_box_reid, pid_inputs, pid_label, pid_logits


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
