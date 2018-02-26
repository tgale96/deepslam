import torch

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


# input dims
batch_size, in_dims, hidden_dims, out_dims = 64, 1000, 100, 10

model = PoseNet(in_dims, hidden_dims, out_dims)














