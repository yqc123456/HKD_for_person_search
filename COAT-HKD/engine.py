# This file is part of COAT, and is distributed under the
# OSI-approved BSD 3-Clause License. See top-level LICENSE file or
# https://github.com/Kitware/COAT/blob/master/LICENSE for details.

import math
import sys
from copy import deepcopy

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from eval_func import eval_detection, eval_search_cuhk, eval_search_prw
from utils.utils import MetricLogger, SmoothedValue, mkdir, reduce_dict, warmup_lr_scheduler
from utils.transforms import mixup_data
from utils.utils import compare_logits


def to_device(images, targets, device):
    images = [image.to(device) for image in images]
    for t in targets:
        t["boxes"] = t["boxes"].to(device)
        t["labels"] = t["labels"].to(device)
    return images, targets


def train_one_epoch(cfg, model, optimizer, data_loader, device, epoch, tfboard, softmax_criterion_s2,
                    softmax_criterion_s3, softmax_criterion_s2_teacher, softmax_criterion_s3_teacher):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    # warmup learning rate in the first epoch
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = len(data_loader) - 1
        warmup_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for i, (images, targets) in enumerate(
            metric_logger.log_every(data_loader, cfg.DISP_PERIOD, header)
    ):
        images, targets = to_device(images, targets, device)

        # if using image based data augmentation
        if cfg.INPUT.IMAGE_MIXUP:
            images = mixup_data(images, alpha=0.8)

        loss_dict, feat_det2, label_det2, feat_det3, label_det3, feat_ins2, label_ins2, feat_ins3, label_ins3 = model(
            images, targets)

        if cfg.MODEL.LOSS.USE_SOFTMAX:
            softmax_loss_2nd, logits_cedet2 = softmax_criterion_s2(feat_det2, label_det2)
            softmax_loss_3rd, logits_cedet3 = softmax_criterion_s3(feat_det3, label_det3)
            loss_dict.update(loss_ce_2nd=softmax_loss_2nd * cfg.SOLVER.LW_RCNN_SOFTMAX_2ND)
            loss_dict.update(loss_ce_3rd=softmax_loss_3rd * cfg.SOLVER.LW_RCNN_SOFTMAX_3RD)

            if cfg.lw_celoss_reid > 0:
                if cfg.use_reid_head2:
                    softmax_loss_2nd_teacher, logits_ceins2 = softmax_criterion_s2_teacher(feat_ins2, label_ins2)
                    loss_dict.update(loss_ce_2nd_teac=softmax_loss_2nd_teacher * cfg.lw_celoss_reid)
                if cfg.use_reid_head3:
                    softmax_loss_3rd_teacher, logits_ceins3 = softmax_criterion_s3_teacher(feat_ins3, label_ins3)
                    loss_dict.update(loss_ce_3rd_teac=softmax_loss_3rd_teacher * cfg.lw_celoss_reid)

            if i % 40 == 0 and cfg.lw_celoss_reid > 0:
                if cfg.use_reid_head2:
                    compare_logits(logits_cedet2, logits_ceins2, label_det2, label_ins2, 'ce 2')
                if cfg.use_reid_head3:
                    compare_logits(logits_cedet3, logits_ceins3, label_det3, label_ins3, 'ce 3')

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if cfg.SOLVER.CLIP_GRADIENTS > 0:
            clip_grad_norm_(model.parameters(), cfg.SOLVER.CLIP_GRADIENTS)
        optimizer.step()

        if epoch == 0:
            warmup_scheduler.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if tfboard:
            iter = epoch * len(data_loader) + i
            for k, v in loss_dict_reduced.items():
                tfboard.add_scalars("train", {k: v}, iter)


@torch.no_grad()
def evaluate_performance(
        model, gallery_loader, query_loader, device, use_gt=False, use_cache=False, use_cbgm=False, gallery_size=100):
    """
    Args:
        use_gt (bool, optional): Whether to use GT as detection results to verify the upper
                                bound of person search performance. Defaults to False.
        use_cache (bool, optional): Whether to use the cached features. Defaults to False.
        use_cbgm (bool, optional): Whether to use Context Bipartite Graph Matching algorithm.
                                Defaults to False.
    """
    model.eval()
    if use_cache:
        eval_cache = torch.load("data/eval_cache/eval_cache.pth")
        gallery_dets = eval_cache["gallery_dets"]
        gallery_feats = eval_cache["gallery_feats"]
        query_dets = eval_cache["query_dets"]
        query_feats = eval_cache["query_feats"]
        query_box_feats = eval_cache["query_box_feats"]
    else:
        gallery_dets, gallery_feats = [], []
        for images, targets in tqdm(gallery_loader, ncols=0):
            images, targets = to_device(images, targets, device)
            # if targets[0]['img_name'][0] == 'S':  # 跳过监控图像的检测和全图特征提取过程
            #     continue
            if not use_gt:
                outputs = model(images)
            else:
                boxes = targets[0]["boxes"]
                n_boxes = boxes.size(0)
                embeddings = model(images, targets)
                outputs = [
                    {
                        "boxes": boxes,
                        "embeddings": torch.cat(embeddings),
                        "labels": torch.ones(n_boxes).to(device),
                        "scores": torch.ones(n_boxes).to(device),
                    }
                ]

            for output in outputs:
                box_w_scores = torch.cat([output["boxes"], output["scores"].unsqueeze(1)], dim=1)
                gallery_dets.append(box_w_scores.cpu().numpy())
                gallery_feats.append(output["embeddings"].cpu().numpy())

        # regarding query image as gallery to detect all people
        # i.e. query person + surrounding people (context information)
        query_dets, query_feats = [], []
        for images, targets in tqdm(query_loader, ncols=0):
            images, targets = to_device(images, targets, device)
            # targets will be modified in the model, so deepcopy it
            outputs = model(images, deepcopy(targets), query_img_as_gallery=True)

            # consistency check
            gt_box = targets[0]["boxes"].squeeze()

            assert (
                           gt_box - outputs[0]["boxes"][0]
                   ).sum() <= 0.001, "GT box must be the first one in the detected boxes of query image"

            for output in outputs:
                box_w_scores = torch.cat([output["boxes"], output["scores"].unsqueeze(1)], dim=1)
                query_dets.append(box_w_scores.cpu().numpy())
                query_feats.append(output["embeddings"].cpu().numpy())

        # extract the features of query boxes
        query_box_feats = []
        for images, targets in tqdm(query_loader, ncols=0):
            images, targets = to_device(images, targets, device)
            embeddings = model(images, targets)
            assert len(embeddings) == 1, "batch size in test phase should be 1"
            query_box_feats.append(embeddings[0].cpu().numpy())

        mkdir("data/eval_cache")
        save_dict = {
            "gallery_dets": gallery_dets,
            "gallery_feats": gallery_feats,
            "query_dets": query_dets,
            "query_feats": query_feats,
            "query_box_feats": query_box_feats,
        }
        torch.save(save_dict, "data/eval_cache/eval_cache.pth")

    eval_detection(gallery_loader.dataset, gallery_dets, det_thresh=0.01)
    eval_search_func = (
        eval_search_cuhk if gallery_loader.dataset.name == "CUHK-SYSU" else eval_search_prw
    )
    eval_search_func(
        gallery_loader.dataset,
        query_loader.dataset,
        gallery_dets,
        gallery_feats,
        query_box_feats,
        query_dets,
        query_feats,
        cbgm=use_cbgm,
        gallery_size=gallery_size,
    )