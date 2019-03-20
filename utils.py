def lossCustom(pred, targets):

    # loss functions used
    MSE = MSELoss()
    BCE = BCELoss()
    CE = nn.CrossEntropyLoss()

    # targets
    pt = torch.FloatTensor
    tx, ty, th tw, tcls, tconf = pt([0]), pt([0]), pt([0]), pt([0])

    tx, ty, th, tw, tcls, tconf= targets

    for image in pred:

        lx += MSE(torch.sigmoid(pi[..., 0:2]), txy[i])


def IOUCalc(bb1, bb2):
    # needs to return best anchor's index
    pass


def getTargets(model, targets, pred):
    # targets [image in batch, cls, tx, ty, th, tw]
    # finds best anchor box to use based in IOU of img bbox and prior anchor box dimension

    gridSize= model.module_list[layer][0].nG  # grid size
    anchors =  model.module_list[layer][0].anchor_vec # anchors from yolo layer in cfg

    # Find out which yolo layer prior anchor is best by calculating IOUs
    gws = targets[:, 4] * gridSize
    ghs = targets[:, 5] * gridSize
    bestIdx, _ = IOUCalc(gws, ghs, anchors) # takes the gw and gh of a target box and sees what's the best anchor to use

    # get x, y in terms of grid
    gxs = targets[:, 2] * gridSize
    gys = targets[:, 3] * gridSize

    # get the indices
    gis = gxs.long().t()
    gjs = gys.long().t()

    txs = gxs - gxs.floor() # get it's position with respect to the grid cell
    tys = gys - gys.floor()

    tws = torch.log(gws / anchors[bestIdx])
    ths = torch.log(ghs / anchors[bestIdx])

    imgNum, cls = targets[:, 0:2].long().t()  # target image, target class
    indicies = [imgNum, cls, gis, gjs]

    return txs, tys, ths, tws, indicies  # [img in batch, cls, grid i, grid j]


def customLoss():
