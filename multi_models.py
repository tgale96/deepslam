import torch
from torch import nn

class SlamNet(nn.Module):
    def __init__(self, seq_len = 2, hidden_dims = 1000):
        super(SlamNet, self).__init__()
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2), # b, 128, 239, 319
            nn.ReLU(True),
            nn.MaxPool2d(3, 2), # b, 16, 119, 159
            nn.Conv2d(128, 64, 3, 2), # b, 64, 59, 79
            nn.ReLU(True),
            nn.MaxPool2d(3, 2) # b, 64, 29, 39
        )

        # Depth regression stem
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(seq_len*64, seq_len*64, 3, 2), # b, 64, 59, 79
            nn.ReLU(True),
            nn.ConvTranspose2d(seq_len*64, 128, 3, 2), # b, 128, 119, 159
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 128, 3, 2), # b, 128, 239, 319
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, 4, 2) # b, 1, 480, 640
        )
        
        # Pose regression stem
        self.linear1 = nn.Linear(seq_len*64*29*39, hidden_dims)
        self.relu = nn.ReLU(True)
        self.linear2 = nn.Linear(hidden_dims, 7)

    # TODO(tgale): incorporate the previous image into the
    # pose prediction. Can we also incorporate it for depth?
    # 1. could do input feeding with depth-map predictions
    # 2. could use RNN. How does this effect compute efficiency?
    def forward(self, x):
        # x is a tensor of (N, T, C, H, W)
        # we want to condense N/T for passing
        # through the feature extractor
        N, T, C, H, W = x.shape
        x = x.view(N*T, C, H, W)

        # Apply feature extractor
        enc = self.encoder(x)
        
        # Reshape so that image pairs are 
        # concatenated along their channels
        _, C, H, W = enc.shape
        enc = enc.view(N, T*C, H, W)

        # Decoder for depth prediction
        dec = self.decoder(enc)
        
        # Stem for pose regression
        enc = enc.view(N, -1)
        out = self.linear1(enc)
        out = self.relu(out)
        out = self.linear2(out)

        # PoseNet loss
        xyz = out[:, 0:3]
        wpqr = out[:, 3:]
        norm = torch.norm(wpqr)
        wpqr_norm = wpqr / norm
        return dec, xyz, wpqr_norm
