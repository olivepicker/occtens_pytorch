from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
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
        if (classes is 'present' and fg.sum() == 0):
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

# geoscal, semscal Loss
# https://github.com/astra-vision/MonoScene/blob/master/monoscene/loss/ssc_loss.py
def geo_scal_loss(pred, ssc_target, ignore_index=0, eps=1e-5):
    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)

    # Compute empty and nonempty probabilities
    empty_probs = pred[:, 0, :, :, :]
    nonempty_probs = 1 - empty_probs
    nonempty_probs = torch.clamp(nonempty_probs, min=1e-7, max=1.0 - 1e-7)

    # Remove unknown voxels
    mask = ssc_target != ignore_index
    nonempty_target = ssc_target != 0
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / (nonempty_probs.sum())
    recall = intersection / (nonempty_target.sum())
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / ((1 - nonempty_target).sum())
    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
        + F.binary_cross_entropy(recall, torch.ones_like(recall))
        + F.binary_cross_entropy(spec, torch.ones_like(spec))
    )

def sem_scal_loss(pred, ssc_target, ignore_index=0, n_classes=18):
    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    mask = ssc_target != ignore_index
    for i in range(n_classes):
        # Get probability of class i
        p = pred[:, i, :, :, :]
        
        # Remove unknown voxels
        target_ori = ssc_target
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        completion_target_ori = torch.ones_like(target_ori).float()
        completion_target_ori[target_ori != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0
            if torch.sum(p) > 0:
                precision = nominator / (torch.sum(p))
                loss_precision = F.binary_cross_entropy(
                    precision, torch.ones_like(precision)
                )
                loss_class += loss_precision
            if torch.sum(completion_target) > 0:
                recall = nominator / (torch.sum(completion_target))
                loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                loss_class += loss_recall
            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                    torch.sum(1 - completion_target)
                )
                loss_specificity = F.binary_cross_entropy(
                    specificity, torch.ones_like(specificity)
                )
                loss_class += loss_specificity
            loss += loss_class
    return loss / count

# Scene Tokenizer Loss Wrapper

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

        logits_3d = logits.view(B, Z, C, Y, X).permute(0, 2, 1, 3, 4).contiguous()
        L_ce = self.ce_loss(logits_3d, target)

        logits_2d = logits_3d.permute(0, 2, 1, 3, 4).contiguous()   # (B, Z, C, Y, X)
        logits_2d = logits_2d.view(B * Z, C, Y, X)

        target_2d = target.view(B * Z, Y, X)                        # (B*Z, Y, X)

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