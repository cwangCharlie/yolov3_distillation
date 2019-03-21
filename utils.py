import glob
import random
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from datasets import *


# Set printoptions
torch.set_printoptions(linewidth=1320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5



def lossCustom(pred, targets):
    # loss functions used
    MSE = nn.MSELoss()
    BCE = nn.BCELoss()
    CE = nn.CrossEntropyLoss()
    pt = torch.FloatTensor
    loss, tx, ty, th, tw, tcls, tconf,  = pt([0]), pt([0]), pt([0]), pt([0]), pt([0]), pt([0]), pt([0])

    txs, tys, ths, tws, imgNum, tconf, bestAnchors, tcls, gis, gjs =  targets


    pr = pred[imgNum, bestAnchors, gis, gjs]  # relevant predictions,  closest to anchors

    # make sure there are labeled objects in the image
    if(len(imgNum) > 0 ):
        lx = MSE(torch.sigmoid(pr[..., 0]), txs)
        ly = MSE(torch.sigmoid(pr[..., 1]), tys)
        lw = MSE(pr[..., 2], tws)
        lh = MSE(pr[..., 3], ths)
        lcls = 0.25 * CE(pr[..., 5:], tcls)

    # because it goes [img, anchors, gi, gj, 85], we are taking the 4th slice here
    lconf = 64 * BCE(torch.sigmoid(pred[..., 4]), tconf)
    loss = lx + ly + lw +lh + lconf + lcls
    return loss

def IOUCalc(w1, h1, box2):
    # Returns the IoU of wh1 to wh2. 1 is gw gh, 2 is anchors

    w2, h2 = box2[0], box2[1]

    # Intersection area
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)

    # Union Area
    union_area = (w2 * h2 + 1e-16) + w1 * h1 - inter_area

    return inter_area / union_area  # iou

def getTargets(model, targets, pred):
    # targets [image in batch, cls, tx, ty, th, tw]

    # find yolo layer index
    for layer, mod in enumerate(model.module_defs):
        if(mod['type'] == 'yolo'):
            break

    gridSize= model.module_list[layer][0].nG  # grid size
    anchors =  model.module_list[layer][0].anchor_vec # anchors from yolo layer in cfg


    # Find out which yolo layer prior anchor is best by calculating IOUs
    gws = targets[:, 4] * gridSize
    ghs = targets[:, 5] * gridSize
    iou = [IOUCalc(gws, ghs, x) for x in anchors]
    _, bestAnchors = torch.stack(iou, 0).max(0)

    # get x, y in terms of grid
    gxs = targets[:, 2] * gridSize
    gys = targets[:, 3] * gridSize

    # get the indices
    gis = gxs.long()
    gjs = gys.long()


    txs = gxs - gxs.floor() # get it's position with respect to the grid cell
    tys = gys - gys.floor()

    tws = torch.log(gws / anchors[bestAnchors,0])
    ths = torch.log(ghs / anchors[bestAnchors,1])


    imgNum, cls = targets[:, 0:2].long().t()  # target image, target class

    # confidence prediction, zero unless object label present
    tconf = torch.zeros_like(pred[..., 0])
    tconf[imgNum, bestAnchors, gis, gjs] = 1 # populate all these with 1's


    return txs, tys, ths, tws, imgNum, tconf, bestAnchors, cls, gis, gjs  # [img in batch, cls, grid i, grid j]
