import argparse
import time

import test  # Import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *

def train(
        cfg,
        data_cfg,
        img_size=416,
        resume=False,
        epochs=100,
        batch_size=16,
        accumulated_batches=1,
        multi_scale=False,
        freeze_backbone=False,
        weightPath,
):

    # Pass in train data configuration
    train_path = parse_data_cfg(data_cfg)['train']

    # Initialize model
    model = Darknet(cfg, img_size)

    # Get dataloader
    dataloader = LoadImagesAndLabels(train_path, batch_size, img_size, multi_scale=multi_scale, augment=True)

    # Parameters
    lr0 = 0.001
    start_epoch = 0


    # Load all incoming weights
    load_darknet_weights(model, weights)


    # Transfer learning (train only YOLO layers)

    # Freeze all but YOLO layer
    # for i, (name, p) in enumerate(model.named_parameters()):
    #     p.requires_grad = True if (p.shape[0] == 255) else False

    # Freeze first X layers
    # for i, (name, p) in enumerate(model.named_parameters()):
    #     if int(name.split('.')[1]) < cutoff:  # if layer < 75

    # optimizer, the filter only passes in the parameter of the model that have grads
    optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=lr0, momentum=.9)

    # Train Loop
    for epoch in range(epochs):
        model.train()
        epoch += 1

        for i, (imgs, targets, _, _) in enumerate(dataloader):

            # Account for empty images
            numTargets = targets.shape[0]
            if(numTargets == 0):
                continue

            pred = model(imgs)

            # txs, tys, ths, tws, indicies [img in batch, cls, grid i, grid j]
            targetsAltered = getTargets(model, targets, pred)
            loss, loss_dict = compute_loss(pred, target_list) # loss = lxy + lwh + lconf + lcls

            # Compute gradient
            loss.backward()

            # Accumulate gradient for x batches before optimizing
            if ((i + 1) % accumulated_batches == 0) or (i == len(dataloader) - 1):
                optimizer.step()
                optimizer.zero_grad()

# Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=270, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--accumulated-batches', type=int, default=1, help='number of batches before optimizer step')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='cfg/coco.data', help='coco.data file path')
    parser.add_argument('--multi-scale', action='store_true', help='random image sizes per batch 320 - 608')
    parser.add_argument('--img-size', type=int, default=32 * 13, help='pixels')
    parser.add_argument('--weights', type=str, help='weights path')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    init_seeds()

    train(
        opt.cfg,
        opt.data_cfg,
        img_size=opt.img_size,
        resume=opt.resume,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        accumulated_batches=opt.accumulated_batches,
        multi_scale=opt.multi_scale,
        weights_path=opt.weights
    )
