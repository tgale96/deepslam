from __future__ import print_function

import argparse
import cv2
from depth_models import DepthNet
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader
from slam_data import SlamDataset

parser = argparse.ArgumentParser()

parser.add_argument("--weights")
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--model", default="DepthNet")
args = parser.parse_args()

# Output dims
h, w, c = 480, 640, 3

# Create the data loaders
test_loader = DataLoader(SlamDataset("data/slam_data.h5", "rgbd_dataset_freiburg1_xyz", False),
                         batch_size = 1, shuffle = False, num_workers = 0)

# Create the model and load the weights
if args.model == "DepthNet":
    model = DepthNet()
else:
    print("Unknown model argument.")
    exit(1)

model.load_state_dict(torch.load(args.weights))
if args.cuda:
    model = model.cuda()
    
criterion = nn.MSELoss(size_average=True)

def test():
    model.eval()
    avg_loss = 0

    for batch_idx, batch in enumerate(test_loader):
        # Get the data for this iter
        data = Variable(batch['rgb'])
        target = Variable(batch['depth'])
        if args.cuda:
            data = data.cuda()
            target = target.cuda()
            
        # Run the forward pass
        dec = model(data)

        depth_map = np.reshape(dec.cpu().data.numpy(), (h, w)) * 5000.
        cv2.imwrite("{}.png".format(batch_idx),
                    depth_map.astype(np.uint16))
        
        # Calculate and print loss
        loss = criterion(dec, target).data[0]
        avg_loss += loss
    
    avg_loss /= len(test_loader)
    print("Test Set:\tAvg. Loss : {:.6f}".format(avg_loss))

test()
