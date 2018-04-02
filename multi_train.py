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
parser.add_argument("--tune", action="store_true")
parser.add_argument("--weights")
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--alpha", type=float, default=10000000)
parser.add_argument("--beta", type=float, default=500)
args = parser.parse_args()

batch_size = 32
alpha = args.alpha
beta = args.beta
epochs = 100
lr = args.lr

# Create the data loaders
train_loader = DataLoader(SlamDataset("data/slam_data.h5", "rgbd_dataset_freiburg1_xyz", 
                                      False, True),
                          batch_size = batch_size, shuffle = True, num_workers = 0)
test_loader = DataLoader(SlamDataset("data/slam_data.h5", "rgbd_dataset_freiburg1_xyz", 
                                     False, True),
                         batch_size = 1, shuffle = False, num_workers = 0)

# Create the model
if args.model == "SlamNet":
    model = SlamNet()
else:
    print("Unknown model argument")
    exit(1)
    
if args.tune:
    model.load_state_dict(torch.load(args.weights))
if args.cuda:
    model = model.cuda()
    
criterion_depth = nn.MSELoss(size_average=True)
criterion_xyz = nn.MSELoss(size_average=True)
criterion_wpqr = nn.MSELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
def train(epoch):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
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
        loss_wpqr = beta * criterion_wpqr(wpqr, target_wpqr)
        loss_pose = loss_xyz + loss_wpqr

        # sum the loss
        loss_total = loss_pose + alpha * loss_depth
        
        # Backprop and update
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        print("Train Epoch: {} [{}/{} ({:.0f}%)]\tDepth Loss: {:.6f}\tPose Loss: {:.6f}".format(
            epoch, batch_idx*batch_size, len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss_depth.data[0], 
            loss_pose.data[0]))
    
def test(epoch):
    model.eval()
    avg_loss_depth = 0
    avg_loss_xyz = 0
    avg_loss_wpqr = 0

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
        loss_wpqr = beta * criterion_wpqr(wpqr, target_wpqr)

        avg_loss_depth += loss_depth.data[0]
        avg_loss_xyz += loss_xyz.data[0]
        avg_loss_wpqr += loss_wpqr.data[0]
    
    avg_loss_depth /= len(test_loader)
    avg_loss_xyz /= len(test_loader)
    avg_loss_wpqr /= len(test_loader)
    print("Test Set:\tAvg. Depth Loss: {:.6f}\tAvg. XYZ Loss: {:.6f}\tAvg. WPQR Loss: {:.6f}"
          .format(avg_loss_depth, avg_loss_xyz, avg_loss_wpqr))

    # Save the model state
    torch.save(model.state_dict(), "depth{}_xyz{}_wpqr{}_epoch_{}_lr{}_b{}.pt"
               .format(avg_loss_depth, avg_loss_xyz, avg_loss_wpqr, epoch, lr, batch_size))
    
for epoch in range(1, epochs+1):
    train(epoch)
    test(epoch)
