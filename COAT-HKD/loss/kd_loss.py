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

        # print('kl loss: ', loss)
        return loss

    def bi_direct(self, logits_s, logits_t):
        p_s = F.log_softmax(logits_s / self.t, dim=1)
        p_t = F.log_softmax(logits_t / self.t, dim=1)
        if logits_s.shape[0] != 0:
            loss = F.kl_div(p_s, p_t, reduction='sum', log_target=True) * (self.t ** 2) / logits_s.shape[0] + \
                   F.kl_div(p_t, p_s, reduction='sum', log_target=True) * (self.t ** 2) / logits_s.shape[0]
        else:
            loss = torch.tensor([0.0]).to(logits_s.device)

        # print('kl loss: ', loss)
        return loss

    def __call__(self, logits_s, logits_t, ):
        if self.bidir:
            loss = self.bi_direct(logits_s, logits_t)
        else:
            loss = self.uni_direct(logits_s, logits_t)
        return loss


def KD_sim_kl_loss(feat_gt, feat_det, useKL=True, temp=4, bidir=True):
    def uni_simkl(logits_t, logits_s):
        p_s = F.log_softmax(logits_s / temp, dim=1)
        p_t = F.softmax(logits_t / temp, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (temp ** 2) / logits_s.shape[0]
        return loss

    def bi_simkl(logits_t, logits_s):
        p_s = F.log_softmax(logits_s / temp, dim=1)
        p_t = F.log_softmax(logits_t / temp, dim=1)

        loss = F.kl_div(p_s, p_t, reduction='sum', log_target=True) * (temp ** 2) / logits_s.shape[0] + \
               F.kl_div(p_t, p_s, reduction='sum', log_target=True) * (temp ** 2) / logits_s.shape[0]

        return loss

    logits_t = feat_gt @ feat_gt.transpose(0, 1)
    logits_s = feat_det @ feat_det.transpose(0, 1)

    if not useKL:
        bs = feat_gt.shape[0]
        logs = F.normalize(logits_s)
        logt = F.normalize(logits_t)
        diff = logt - logs
        loss = (diff * diff).view(-1, 1).sum(0) / (bs * bs)
    else:
        if bidir:
            loss = bi_simkl(logits_t, logits_s)/2
        else:
            loss = uni_simkl(logits_t, logits_s)

    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0).to(feat_gt.device)
    return loss


def KD_feat_loss(feat_reid, feat_srch):
    sim = (feat_reid * feat_srch).sum(1)
    if len(feat_srch) != 0:
        loss = (1 - sim).sum() / len(feat_srch)
    else:
        loss = torch.tensor([0.0]).to(feat_reid.device)

    # loss=sum(  1-feat_reid[i].unsqueeze(dim=0) @ feat_srch[i].unsqueeze(dim=1) for i in range(len(feat_srch)))/ len(feat_srch)
    # print('  kd loss', loss)
    return loss


def KD_sim_loss(feat_gt, feat_det):
    logits_t = feat_gt @ feat_gt.transpose(0, 1)
    logits_s = feat_det @ feat_det.transpose(0, 1)

    temp = 4
    p_s = F.log_softmax(logits_s / temp, dim=1)
    p_t = F.softmax(logits_t / temp, dim=1)
    loss = F.kl_div(p_s, p_t, reduction='sum') * (temp ** 2) / logits_s.shape[0]

    return loss
