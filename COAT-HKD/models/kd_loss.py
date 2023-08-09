import torch.nn.functional as F
import torch
import math


class KLLoss:
    """ KL Divergence"""

    def __init__(self, t=4, bidir=False):
        self.t = t
        self.bidir = bidir

    def uni_direct(self, logits_s, logits_t):
        p_s = F.log_softmax(logits_s / self.t, dim=1)
        p_t = F.softmax(logits_t / self.t, dim=1)
        if logits_s.shape[0] != 0:
            loss = F.kl_div(p_s, p_t, reduction='sum') * (self.t ** 2) / logits_s.shape[0]
        else:
            loss = torch.tensor([0.0]).to(logits_s.device)

        return loss

    def bi_direct(self, logits_s, logits_t):
        p_s = F.log_softmax(logits_s / self.t, dim=1)
        p_t = F.log_softmax(logits_t / self.t, dim=1)
        if logits_s.shape[0] != 0:
            loss = F.kl_div(p_s, p_t, reduction='sum', log_target=True) * (self.t ** 2) / logits_s.shape[0] + \
                   F.kl_div(p_t, p_s, reduction='sum', log_target=True) * (self.t ** 2) / logits_s.shape[0]
        else:
            loss = torch.tensor([0.0]).to(logits_s.device)

        return loss

    def __call__(self, logits_s, logits_t):
        if self.bidir:
            loss = self.bi_direct(logits_s, logits_t)
        else:
            loss = self.uni_direct(logits_s, logits_t)
        return loss


def KD_feat_loss(feat_reid, feat_srch):
    sim = (feat_reid * feat_srch).sum(1)
    if len(feat_srch) != 0:
        loss = (1 - sim).sum() / len(feat_srch)
    else:
        loss = torch.tensor([0.0]).to(feat_reid.device)

    return loss


def KD_sim_kl_loss(feat_gt, feat_det, useKL=True):
    logits_t = feat_gt @ feat_gt.transpose(0, 1)
    logits_s = feat_det @ feat_det.transpose(0, 1)

    if not useKL:
        bs = feat_gt.shape[0]
        logs = F.normalize(logits_s)
        logt = F.normalize(logits_t)
        diff = logt-logs
        loss = (diff*diff).view(-1, 1).sum(0) / (bs * bs)
    else:
        temp = 4
        p_s = F.log_softmax(logits_s / temp, dim=1)
        p_t = F.softmax(logits_t / temp, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (temp ** 2) / logits_s.shape[0]
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0).to(feat_gt.device)
    return loss


def KD_pkt_loss(feat_tech, feat_stu, eps=0.0000001):
    """
    Probabilistic Knowledge Transfer for deep representation learning
    input: L2 normed feat

    """
    # Calculate the cosine similarity
    model_similarity = feat_stu @ feat_stu.transpose(0, 1)
    target_similarity = feat_tech @ feat_tech.transpose(0, 1)

    # Scale cosine similarity to 0..1
    model_similarity = (model_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0

    # Transform them into probabilities
    model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
    target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

    # Calculate the KL-divergence
    loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)))
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0).to(feat_stu.device)
    return loss


def kd_Attention_loss(f_s, f_t, p=2):
    """Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks
    via Attention Transfer
    code: https://github.com/szagoruyko/attention-transfer
    """
    at_s = F.normalize(f_s.pow(p).mean(1).view(f_s.size(0), -1))
    at_t = F.normalize(f_t.pow(p).mean(1).view(f_t.size(0), -1))
    loss = (at_s - at_t).pow(2).mean()
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0).to(f_s.device)
    return loss
