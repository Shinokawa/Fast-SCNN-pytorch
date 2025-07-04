"""Custom losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable

__all__ = ['MixSoftmaxCrossEntropyLoss', 'MixSoftmaxCrossEntropyOHEMLoss', 'DiceLoss', 'MixDiceLoss']


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation tasks like lane detection."""
    
    def __init__(self, smooth=1e-6, **kwargs):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: [N, C, H, W] - predicted logits
            target: [N, H, W] - ground truth labels
        """
        if pred.dim() == 4 and pred.size(1) > 1:
            # Convert to probabilities and take the positive class
            pred = F.softmax(pred, dim=1)[:, 1, :, :]  # Take class 1 (lane)
        elif pred.dim() == 4 and pred.size(1) == 1:
            pred = torch.sigmoid(pred.squeeze(1))
        
        # Flatten
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1).float()
        
        # Dice coefficient
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class MixDiceLoss(nn.Module):
    """Mixed Dice Loss with auxiliary loss support."""
    
    def __init__(self, aux=True, aux_weight=0.4, smooth=1e-6, **kwargs):
        super(MixDiceLoss, self).__init__()
        self.aux = aux
        self.aux_weight = aux_weight
        self.dice_loss = DiceLoss(smooth=smooth)
    
    def forward(self, preds, target):
        """
        Args:
            preds: tuple of predictions (main_pred, aux_pred) or single prediction
            target: ground truth labels
        """
        if isinstance(preds, tuple):
            main_pred = preds[0]
            loss = self.dice_loss(main_pred, target)
            
            if self.aux and len(preds) > 1:
                aux_pred = preds[1]
                aux_loss = self.dice_loss(aux_pred, target)
                loss += self.aux_weight * aux_loss
                
            return loss
        else:
            return self.dice_loss(preds, target)


class FocalDiceLoss(nn.Module):
    """Combined Focal + Dice Loss for handling class imbalance."""
    
    def __init__(self, alpha=0.5, gamma=2.0, dice_weight=0.5, smooth=1e-6, **kwargs):
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss(smooth=smooth)
    
    def focal_loss(self, pred, target):
        """Focal loss component."""
        if pred.dim() == 4 and pred.size(1) > 1:
            # Cross entropy for multi-class
            ce_loss = F.cross_entropy(pred, target, reduction='none')
            pt = torch.exp(-ce_loss)
        else:
            # Binary case
            pred_prob = torch.sigmoid(pred.squeeze(1))
            target_float = target.float()
            ce_loss = F.binary_cross_entropy(pred_prob, target_float, reduction='none')
            pt = torch.where(target_float == 1, pred_prob, 1 - pred_prob)
        
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
    
    def forward(self, pred, target):
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return (1 - self.dice_weight) * focal + self.dice_weight * dice


class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.2, ignore_label=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=ignore_label)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return self._aux_forward(*inputs)
        else:
            return super(MixSoftmaxCrossEntropyLoss, self).forward(*inputs)


class SoftmaxCrossEntropyOHEMLoss(nn.Module):
    def __init__(self, ignore_label=-1, thresh=0.7, min_kept=256, use_weight=True, **kwargs):
        super(SoftmaxCrossEntropyOHEMLoss, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            print("w/ class balance")
            weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
                                        1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                                        1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)
        else:
            print("w/o class balance")
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    def forward(self, predict, target, weight=None):
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))

        n, c, h, w = predict.size()
        input_label = target.data.cpu().numpy().ravel().astype(np.int32)
        x = np.rollaxis(predict.data.cpu().numpy(), 1).reshape((c, -1))
        input_prob = np.exp(x - x.max(axis=0).reshape((1, -1)))
        input_prob /= input_prob.sum(axis=0).reshape((1, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if self.min_kept >= num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = pred.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if pred[threshold_index] > self.thresh:
                    threshold = pred[threshold_index]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        valid_flag_new = input_label != self.ignore_label
        # print(np.sum(valid_flag_new))
        target = Variable(torch.from_numpy(input_label.reshape(target.size())).long().cuda())

        return self.criterion(predict, target)


class MixSoftmaxCrossEntropyOHEMLoss(SoftmaxCrossEntropyOHEMLoss):
    def __init__(self, aux=False, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyOHEMLoss, self).__init__(ignore_label=ignore_index, **kwargs)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return self._aux_forward(*inputs)
        else:
            return super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(*inputs)
