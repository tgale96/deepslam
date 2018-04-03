from __future__ import print_function

import argparse
import cv2
from multi_models import SlamNet
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader
from slam_data import SlamDataset

parser = argparse.ArgumentParser()

parser.add_argument("--model", default="SlamNet")
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--weights")
parser.add_argument("--dir", default="")
args = parser.parse_args()

# Create the data loaders
test_loader = DataLoader(SlamDataset("data/slam_data.h5", "rgbd_dataset_freiburg1_xyz", 
                                     False, True),
                         batch_size = 1, shuffle = False, num_workers = 0)

# Output resolution
h, w = 480, 640

# Create the model
if args.model == "SlamNet":
    model = SlamNet()
else:
    print("Unknown model argument")
    exit(1)

# Load the weights
model.load_state_dict(torch.load(args.weights))
if args.cuda:
    model = model.cuda()
    
criterion_depth = nn.MSELoss(size_average=True)
criterion_xyz = nn.MSELoss(size_average=True)
criterion_wpqr = nn.MSELoss(size_average=True)

def to_pose_str(xyz, wpqr):
    xyz = [str(x) for x in list(xyz.cpu().data.numpy()[0, :])]
    wpqr = [str(x) for x in list(wpqr.cpu().data.numpy()[0, :])]
    return  " ".join(xyz) + " " + " ".join(wpqr) + "\n"

def test():
    model.eval()
    avg_loss_depth = 0
    avg_loss_xyz = 0
    avg_loss_wpqr = 0

    pose_out = open(args.dir + "pose.txt", 'w')
    
    for batch_idx, batch in enumerate(test_loader):
        # Get the data for this iter
        data = Variable(batch['rgb_pair'])
        target_depth = Variable(batch['depth'])
        target_pose = Variable(batch['pose_diff'])
        if args.cuda:
            data = data.cuda()
            target_depth = target_depth.cuda()
            target_pose = target_pose.cuda()
            
        target_xyz = target_pose[:, 0:3]
        target_wpqr = target_pose[:, 3:]

        # Run the forward pass
        depth, xyz, wpqr = model(data)
        
        # Calculate depth loss
        loss_depth = criterion_depth(depth, target_depth)
        
        # Calculate pose loss
        loss_xyz = criterion_xyz(xyz, target_xyz)
        loss_wpqr = criterion_wpqr(wpqr, target_wpqr)
        
        avg_loss_depth += loss_depth.data[0]
        avg_loss_xyz += loss_xyz.data[0]
        avg_loss_wpqr += loss_wpqr.data[0]

        # Write results
        pose_out.write(to_pose_str(xyz, wpqr))

        # Clamp negative predictions to 0
        depth_map = np.reshape(depth.cpu().data.numpy(), (h, w)) * 5000.
        depth_map[depth_map < 0.0] = 0.0
        cv2.imwrite(args.dir + "{}.png".format(batch_idx),
                    depth_map.astype(np.uint16))
        
    avg_loss_depth /= len(test_loader)
    avg_loss_xyz /= len(test_loader)
    avg_loss_wpqr /= len(test_loader)
    print("Test Set:\tAvg. Depth Loss: {:.6f}\tAvg. XYZ Loss: {:.6f}\tAvg. WPQR Loss: {:.6f}"
          .format(avg_loss_depth, avg_loss_xyz, avg_loss_wpqr))

test()
