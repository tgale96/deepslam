from __future__ import print_function

import argparse
import torch
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader
from slam_data import SlamDataset

parser = argparse.ArgumentParser(
    description="script to train a small pose predictor")

parser.add_argument("--write_results", action="store_true")
parser.add_argument("--cuda", action="store_true")
args = parser.parse_args()
    
class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2), # b, 16, 239, 319
            nn.ReLU(True),
            nn.MaxPool2d(3, 2), # b, 16, 119, 159
            nn.Conv2d(16, 8, 3, 2), # b, 8, 59, 79
            nn.ReLU(True),
            nn.MaxPool2d(3, 2) # b, 8, 29, 39
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 8, 3, 2), # b, 8, 59, 79
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 16, 3, 2), # b, 16, 119, 159
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 16, 3, 2), # b, 16, 239, 319
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 4, 2) # b, 1, 480, 640
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Output dims
h, w, c = 480, 640, 3
batch_size = 32
epochs = 100
lr = 0.00000000001

# Create the data loaders
train_loader = DataLoader(SlamDataset("data/slam_data.h5", "rgbd_dataset_freiburg1_xyz", False),
                          batch_size = batch_size, shuffle = True, num_workers = 0)
test_loader = DataLoader(SlamDataset("data/slam_data.h5", "rgbd_dataset_freiburg1_xyz", False),
                         batch_size = 1, shuffle = False, num_workers = 0)

# Create the model
model = DepthNet()
if args.cuda:
    model = model.cuda()
    
criterion = nn.MSELoss(size_average=True)
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
        loss /= h*w*c
        
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

    if args.write_results:
        out = open("eval_{}.txt".format(epoch), 'w')
        
    for batch_idx, batch in enumerate(test_loader):
        # Get the data for this iter
        data = Variable(batch['rgb'])
        target = Variable(batch['depth'])
        if args.cuda:
            data = data.cuda()
            target = target.cuda()
            
        # Run the forward pass
        dec = model(data)

        # Write results to file
        if args.write_results:
            print("Result writing not supported")
            exit(1)
            out.write("")
                
        # Calculate and print loss
        loss = criterion(dec, target).data[0]
        loss /= h*w*c
        avg_loss += loss
    
    avg_loss /= len(test_loader)
    print("Test Set:\tAvg. Loss : {:.6f}".format(avg_loss))

    # Save the model state
    torch.save(model.state_dict(), "mse{}_epoch_{}_lr{}_b{}.pt"
               .format(avg_loss, epoch, lr, batch_size))

    
for epoch in range(1, epochs+1):
    train(epoch)
    test(epoch)
