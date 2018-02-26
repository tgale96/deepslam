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
        return out

# Input dims
batch_size, in_dims, hidden_dims, out_dims = 32, 480*640*3, 100, 7
epochs = 1

# Create the data loader
dataloader = DataLoader(SlamDataset("data/slam_data.h5", "rgbd_dataset_freiburg1_xyz"),
                        batch_size = batch_size, shuffle = True, num_workers = 1)

# Create the model
model = PoseNet(in_dims, hidden_dims, out_dims)

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.000000000000001)

for e in range(epochs):
    for i_batch, batch in enumerate(dataloader):
        # Run the forward pass
        data = Variable(batch['rgb'])
        target = Variable(batch['pose'])
        output = model(data)

        # Calculate and print loss
        loss = criterion(output, target)
        print(i_batch, loss.data[0])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()











