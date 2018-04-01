from torch import nn

class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2), # b, 128, 239, 319
            nn.ReLU(True),
            nn.MaxPool2d(3, 2), # b, 16, 119, 159
            nn.Conv2d(128, 64, 3, 2), # b, 64, 59, 79
            nn.ReLU(True),
            nn.MaxPool2d(3, 2) # 64, 8, 29, 39
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, 2), # b, 64, 59, 79
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 128, 3, 2), # b, 128, 119, 159
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 128, 3, 2), # b, 128, 239, 319
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, 4, 2) # b, 1, 480, 640
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
