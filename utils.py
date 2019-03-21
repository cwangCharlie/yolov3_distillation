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


# Set printoptions
torch.set_printoptions(linewidth=1320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5



def lossCustom(pred, targets):
    # loss functions used
    MSE = nn.MSELoss()
    BCE = nn.BCELoss()
    CE = nn.CrossEntropyLoss()
    tx, ty, tw, th, tcls, tconf, indices = targets
    b, a, gj, gi = indices[i]  # image, anchor, gridx, gridy
    # targets
    pt = torch.FloatTensor
    tx, ty, th, tw, tcls, tconf = pt([0]), pt([0]), pt([0]), pt([0]), pt([0])
    pr = pred[b, a, gj, gi]  # relevant predictions,  closest to anchors

    # make sure there are labeled objects in the image
    if(len(indices[0]) > 0 ):
        lx = MSE(torch.sigmoid(pr[..., 0]), tx)
        lx = MSE(torch.sigmoid(pr[..., 1]), ty)
        lw = MSE(pr[..., 2], tw)
        lh = MSE(pr[..., 3], th)
        lcls = 0.25 * CE(pr[..., 5:], tcls)

    lconf = 64 * BCE(pr[..., 4], tconf)
    loss = lxy + lwh + lconf  + lcls

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
    # finds best anchor box to use based in IOU of img bbox and prior anchor box dimension

    layer = get_yolo_layers(model)
    layer = layer[0]

    gridSize= model.module_list[layer][0].nG  # grid size
    anchors =  model.module_list[layer][0].anchor_vec # anchors from yolo layer in cfg

    print("targets:", targets.size())

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
    print(cls)
    exit()
    # indicies = [imgNum, cls, gis, gjs]

    return txs, tys, ths, tws, imgNum, cls, gis, gjs  # [img in batch, cls, grid i, grid j]

def float3(x):  # format floats to 3 decimals
    return float(format(x, '.3f'))
