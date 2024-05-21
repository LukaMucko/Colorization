import torch
import torch.nn as nn
import torch.nn.functional as F
"""
May work if initialized correctly!
"""

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, channels, stride=1, first=True):
        super(ResidualBlock, self).__init__()

        self.net = nn.Sequential(nn.Conv2d(in_channels, channels, 3, padding=1, stride=stride, bias=False),
                                 nn.InstanceNorm2d(channels),
                                 nn.ReLU(True),
                                 nn.Conv2d(channels, channels, 3, stride=1, bias=False, padding=1),
                                 nn.InstanceNorm2d(channels),
                                 nn.ReLU(True),
                                 nn.Conv2d(channels, out_channels, 1, bias=False),
                                 )

        if first:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                                            nn.InstanceNorm2d(out_channels))
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        out = self.net(x)
        out = out + self.downsample(x)
        return F.relu(out)
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1) #(n-1)s - 2p + k = 2n
        self.resblock = ResidualBlock(in_channels+out_channels, out_channels, out_channels, first=True)
    
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.resblock(x)
        return F.relu(x)
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.resblock = ResidualBlock(in_channels, out_channels, out_channels, first=True)
        self.maxpool = nn.MaxPool2d(2)
        
    def forward(self, x):
        x = self.resblock(x)
        return self.maxpool(x)

class Generator(nn.Module):
    def __init__(self, input_channel, output_channel, image_size=256):
        super().__init__()
        assert image_size % 2**8 == 0, 'image size must be a multiple of 256'
        
        self.initial = ResidualBlock(input_channel, 64, 64, first=True)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.down5 = Down(512, 512)
        
        self.up1 = Up(512, 512)
        self.up2 = Up(512, 512)
        self.up3 = Up(512, 256)
        self.up4 = Up(256, 128)
        self.up5 = Up(128, 64)
        
        self.final = nn.Conv2d(64, output_channel, 3, padding=1)
        self.tanh = nn.Tanh()
        
        
    def forward(self, x): 
        
        # Downsampling
        x1 = self.initial(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        
        # Upsampling
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
                
        x = self.final(x)
        
        return self.tanh(x)

        
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, drip=0.2, patch=True):
        super().__init__()
        self.drip=drip
        self.patch = patch
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(drip),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(drip),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(drip),
            
            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(drip),
            
            nn.Conv2d(512, 1024, 5, stride=2, padding=1, bias=False),
                        
            nn.AdaptiveAvgPool2d(1)
        )
        if not patch:
            self.fc = nn.LazyLinear(1)

    def forward(self, l, ab):
        x = torch.cat([l, ab], dim=1)
        x = self.conv_layers(x)
        if not self.patch:
            x = torch.flatten(x, 1)
            return self.fc(x)
        
        return x

def init_weights(model, method='normal', std=0.02):
    if method == 'normal':
        for m in model.parameters():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                torch.nn.init.normal_(m.weight, 0.0, std)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
