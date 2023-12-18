import torch
import torch.nn as nn
import torch.nn.functional as F
from .soft_skeleton import soft_skel,counter
from torch.autograd import Variable
import cv2
import numpy as np

class soft_cldice(nn.Module):
    def __init__(self, iter_=3, smooth = 1.):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(self,y_pred, y_true):
        y_true = torch.sigmoid(y_pred)
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        tprec = (torch.sum(torch.mul(skel_pred, y_true)[:,:,:,:])+self.smooth)/(torch.sum(skel_pred[:,:,:,:])+self.smooth)
        tsens = (torch.sum(torch.mul(skel_true, y_pred)[:,:,:,:])+self.smooth)/(torch.sum(skel_true[:,:,:,:])+self.smooth)
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice


def soft_dice(y_pred,y_true):
    """[function to compute dice loss]

    Args:
        y_true ([float32]): [ground truth image]
        y_pred ([float32]): [predicted image]

    Returns:
        [float32]: [loss value]
    """
    smooth = 1
    intersection = torch.sum((y_true * y_pred)[:,:,:,:])
    coeff = (2. *  intersection + smooth) / (torch.sum(y_true[:,:,:,:]) + torch.sum(y_pred[:,:,:,:]) + smooth)
    return (1. - coeff)


class soft_dice_cldice(nn.Module):
    def __init__(self, iter_=5, alpha=0.5 ,smooth = 1.):
        super(soft_dice_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha

    def forward(self,y_pred,y_true):

        # dice loss
        dice = soft_dice(y_pred,y_true)

        # cldice loss
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)

        tprec = (torch.sum(torch.mul(skel_pred, y_true)[:,:,:,:])+self.smooth)/(torch.sum(skel_pred[:,:,:,:])+self.smooth)
        tsens = (torch.sum(torch.mul(skel_pred, y_true)[:,:,:,:])+self.smooth)/(torch.sum(skel_true[:,:,:,:])+self.smooth)
        cl_dice = 1. - 2.0 * (tprec * tsens) / (tprec + tsens)

        return (1.0 - self.alpha) * dice + self.alpha * cl_dice





