from __future__ import print_function

import argparse
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from slam_data import SlamDataset

parser = argparse.ArgumentParser(
    description="script to train a small pose predictor")

parser.add_argument("--write_results", action="store_true")
parser.add_argument('--hidden_size', help="num units in hidden layer",
                    type=int, default=100)                    
args = parser.parse_args()
    
class PoseNet(torch.nn.Module):
    def __init__(self, in_dims, hidden_dims, out_dims):
        super(PoseNet, self).__init__()
        self.linear1 = torch.nn.Linear(in_dims, hidden_dims)
        self.relu = torch.nn.ReLU(inplace=True)
        self.linear2 = torch.nn.Linear(hidden_dims, out_dims)
        
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)

        # PoseNet loss
        xyz = out[:, 0:3]
        wpqr = out[:, 3:]
        norm = torch.norm(wpqr)
        wpqr_norm = wpqr / norm
        return xyz, wpqr_norm

# Input dims
batch_size, in_dims, hidden_dims, out_dims = 32, 480*640*3, args.hidden_size, 7
epochs = 100
beta = 500
lr = 0.00000001

# Create the data loaders
train_loader = DataLoader(SlamDataset("data/slam_data.h5", "rgbd_dataset_freiburg1_xyz"),
                          batch_size = batch_size, shuffle = True, num_workers = 0)
test_loader = DataLoader(SlamDataset("data/slam_data.h5", "rgbd_dataset_freiburg1_xyz"),
                         batch_size = 1, shuffle = False, num_workers = 0)

# Create the model
model = PoseNet(in_dims, hidden_dims, out_dims)

criterion_xyz = torch.nn.MSELoss(size_average=True)
criterion_wpqr = torch.nn.MSELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

def train(epoch):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        # Get the data for this iter
        data = Variable(batch['rgb'])
        target = Variable(batch['pose'])
        target_xyz = target[:, 0:3]
        target_wpqr = target[:, 3:]
        
        # Run the forward pass
        xyz, wpqr = model(data)
        
        # Calculate and print loss
        loss_xyz = criterion_xyz(xyz, target_xyz)
        loss_wpqr = beta * criterion_wpqr(wpqr, target_wpqr)
        loss = sum([loss_xyz, loss_wpqr])
        
        # Backprop and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
            epoch, batch_idx*batch_size, len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.data[0]))

def to_pose_str(xyz, wpqr):
    xyz = [str(x) for x in list(xyz.data.numpy()[0, :])]
    wpqr = [str(x) for x in list(wpqr.data.numpy()[0, :])]
    return  " ".join(xyz) + " " + " ".join(wpqr) + "\n"
    
def test(epoch):
    model.eval()
    avg_loss_xyz = 0
    avg_loss_wpqr = 0

    if args.write_results:
        out = open("eval_{}.txt".format(epoch), 'w')
        
    for batch_idx, batch in enumerate(test_loader):
        # Get the data for this iter
        data = Variable(batch['rgb'])
        target = Variable(batch['pose'])
        target_xyz = target[:, 0:3]
        target_wpqr = target[:, 3:]

        # Run the forward pass
        xyz, wpqr = model(data)

        # Write results to file
        if args.write_results:
            out.write(to_pose_str(xyz, wpqr))
                
        # Calculate and print loss. Don't scale by beta
        avg_loss_xyz += criterion_xyz(xyz, target_xyz).data[0]
        avg_loss_wpqr += criterion_wpqr(wpqr, target_wpqr).data[0]
    
    avg_loss_xyz /= len(test_loader)
    avg_loss_wpqr /= len(test_loader)
    print("Test Set:\tAvg. Loss (XYZ): {:.6f}\tAvg. Loss (WPQR): {:.6f}"
          .format(avg_loss_xyz, avg_loss_wpqr))

    # Save the model state
    torch.save(model.state_dict(), "xyz{}_wpqr{}_epoch_{}_h{}_lr{}_b{}_beta{}.pt"
               .format(avg_loss_xyz, avg_loss_wpqr, epoch,
                       hidden_dims, lr, batch_size, beta))

    
for epoch in range(1, epochs+1):
    train(epoch)
    test(epoch)
