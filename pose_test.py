
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from slam_data import SlamDataset

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
batch_size, in_dims, hidden_dims, out_dims = 32, 480*640*3, 1000, 7
epochs = 100
beta = 500

# Create the data loader
dataloader = DataLoader(SlamDataset("data/slam_data.h5", "rgbd_dataset_freiburg1_xyz"),
                        batch_size = batch_size, shuffle = True, num_workers = 0)

# Create the model
model = PoseNet(in_dims, hidden_dims, out_dims)

criterion_xyz = torch.nn.MSELoss(size_average=True)
criterion_wpqr = torch.nn.MSELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.00000001)

for e in range(epochs):
    for i_batch, batch in enumerate(dataloader):
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
        print(i_batch, loss.data[0])

        # Backprop and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
