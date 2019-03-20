def lossCustom(pred, targets):
    # loss functions used
    MSE = MSELoss()
    BCE = BCELoss()
    CE = nn.CrossEntropyLoss()
    tx, ty, tw, th, tcls, tconf, indices = targets
    b, a, gj, gi = indices[i]  # image, anchor, gridx, gridy
    # targets
    pt = torch.FloatTensor
    tx, ty, th tw, tcls, tconf = pt([0]), pt([0]), pt([0]), pt([0])
    pr = pred[b, a, gj, gi]  # relevant predictions,  closest to anchors
    make sure there are labeled objects in the image
    if(len(indices[0]) > 0 ):
        lx = MSE(torch.sigmoid(pr[..., 0]), tx)
        lx = MSE(torch.sigmoid(pr[..., 1]), ty)
        lw = MSE(pr[..., 2], tw)
        lh = MSE(pr[..., 3], th)
        lcls = 0.25 * CE(pr[..., 5:], tcls)

    lconf = 64 * BCE(pr[..., 4], tconf)
    loss = lxy + lwh + lconf  + lcls

    return loss


def IOUCalc(bb1, bb2):
    # needs to return best anchor's index
    bb1 + bb2

def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'
    # cutoff: save layers between 0 and cutoff (if cutoff = -1 all are saved)
    weights_file = weights.split(os.sep)[-1]

    # Try to download weights if not available locally
    if not os.path.isfile(weights):
        print(weights + ' not found')

    # Open the weights file
    fp = open(weights, 'rb')
    header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

    # Needed to write header when saving weights
    self.header_info = header

    self.seen = header[3]  # number of images seen during training
    weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
    fp.close()

    ptr = 0
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            if module_def['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w


"""
    @:param path    - path of the new weights file
    @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
"""


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
