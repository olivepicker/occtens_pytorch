from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
from einops import rearrange
import torch.nn.functional as F
import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse

# Lovasz-Softmax Loss
# https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def isnan(x):
    return x != x
    
def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

def geo_scal_loss(pred, ssc_target, ignore_index=0, eps=1e-5):
    pred = F.softmax(pred, dim=1)
    
    empty_probs = pred[:, 17, :, :, :] # free class
    nonempty_probs = 1 - empty_probs
    nonempty_probs = torch.clamp(nonempty_probs, min=1e-7, max=1.0 - 1e-7)

    mask = ssc_target != ignore_index
    nonempty_target = (ssc_target != 0).float()
    
    nonempty_target = nonempty_target[mask]
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    
    prec_denom = nonempty_probs.sum() + eps
    recall_denom = nonempty_target.sum() + eps
    spec_denom = (1 - nonempty_target).sum() + eps
    
    precision = intersection / prec_denom
    recall = intersection / recall_denom
    spec = ((1 - nonempty_target) * empty_probs).sum() / spec_denom

    loss_precision = -torch.log(precision + eps)
    loss_recall = -torch.log(recall + eps)
    loss_spec = -torch.log(spec + eps)

    return loss_precision + loss_recall + loss_spec

def sem_scal_loss(pred, ssc_target, ignore_index=0, n_classes=18, eps=1e-5):
    pred = F.softmax(pred, dim=1)
    mask = ssc_target != ignore_index
    
    p_masked = pred.permute(0, 2, 3, 4, 1)[mask] # (N_valid, C)
    target_masked = ssc_target[mask]             # (N_valid,)

    loss = 0.0
    count = 0.0

    for i in range(n_classes):

        p_i = p_masked[:, i]
        target_i = (target_masked == i).float()
        
        sum_target = torch.sum(target_i)
        sum_p = torch.sum(p_i)
        
        if sum_target < eps and sum_p < eps:
            continue
            
        intersection = torch.sum(p_i * target_i)
        
        loss_class = 0.0
        
        if sum_p > eps:
            precision = intersection / (sum_p + eps)
            precision = torch.clamp(precision, min=eps, max=1.0)
            loss_class += -torch.log(precision)

        if sum_target > eps:
            recall = intersection / (sum_target + eps)
            recall = torch.clamp(recall, min=eps, max=1.0)
            loss_class += -torch.log(recall)

        inverse_target = 1.0 - target_i
        sum_inverse_target = torch.sum(inverse_target)
        
        if sum_inverse_target > eps:
            specificity = torch.sum((1.0 - p_i) * inverse_target) / (sum_inverse_target + eps)
            specificity = torch.clamp(specificity, min=eps, max=1.0)
            loss_class += -torch.log(specificity)

        loss += loss_class
        count += 1.0

    if count > 0:
        return loss / count
    else:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

class CustomSceneLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 18,
        lambda_ce: float = 10.0,
        lambda_lovasz: float = 1.0,
        lambda_geoscal: float = 0.3,
        lambda_semscal: float = 0.5,
        ignore_index: int = 255,
        ce_class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_ce = lambda_ce
        self.lambda_lovasz = lambda_lovasz
        self.lambda_geoscal = lambda_geoscal
        self.lambda_semscal = lambda_semscal
        self.ignore_index = ignore_index

        self.ce_loss = nn.CrossEntropyLoss(
            weight=ce_class_weights,
            ignore_index=ignore_index
        )

    def forward(self, logits, target):
        B, Cz, Y, X = logits.shape
        C = self.num_classes
        Z = Cz // C
        assert Cz == C * Z, f"Channel dim {Cz} != num_classes({C}) * num_z({Z})"

        logits_2d = rearrange(logits, 'b (z c) y x -> b z c y x', c=C).contiguous()
        logits_3d = rearrange(logits_2d, 'b z c y x -> b c z y x').contiguous()
        L_ce = self.ce_loss(logits_3d, target)

        logits_2d = rearrange(logits_2d, 'b z c y x -> (b z) c y x')
        target_2d = rearrange(target, 'b z y x -> (b z) y x')

        probas_2d = F.softmax(logits_2d, dim=1)
        L_lovasz = lovasz_softmax(
            probas_2d,
            target_2d,
            classes='present',
            per_image=False,
            ignore=self.ignore_index
        )

        L_geoscal = geo_scal_loss(
            logits_3d,
            target,
            ignore_index=self.ignore_index
        )

        L_semscal = sem_scal_loss(
            logits_3d,
            target,
            ignore_index=self.ignore_index,
            n_classes = self.num_classes
        )

        loss = (
            self.lambda_ce * L_ce
            + self.lambda_lovasz * L_lovasz
            + self.lambda_geoscal * L_geoscal
            + self.lambda_semscal * L_semscal
        )

        return {
            "loss": loss,
            "loss_ce": L_ce,
            "loss_lovasz": L_lovasz,
            "loss_geoscal": L_geoscal,
            "loss_semscal": L_semscal,
        }