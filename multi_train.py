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
args = parser.parse_args()

batch_size = 32
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
    
depth_criterion = nn.MSELoss(size_average=True)
criterion_xyz = nn.MSELoss(size_average=True)
criterion_wpqr = nn.MSELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
def train(epoch):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        # Get the data for this iter
        data = Variable(batch['rgb'])
        target = Variable(batch['depth'])
        if args.cuda:
            data = data.cuda()
            target = target.cuda()
            
        # Run the forward pass
        dec = model(data)
        
        # Calculate and print loss
        loss = criterion(dec, target)
        
        # Backprop and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
            epoch, batch_idx*batch_size, len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.data[0]))
    
def test(epoch):
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

        # Calculate and print loss
        loss = criterion(dec, target).data[0]
        avg_loss += loss
    
    avg_loss /= len(test_loader)
    print("Test Set:\tAvg. Loss : {:.6f}".format(avg_loss))

    # Save the model state
    torch.save(model.state_dict(), "mse{}_epoch_{}_lr{}_b{}.pt"
               .format(avg_loss, epoch, lr, batch_size))

    
for epoch in range(1, epochs+1):
    train(epoch)
    test(epoch)
