import torch
import torch.nn as nn
import torch.nn.functional as F
"""
May work if initialized correctly!
"""



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, channels, stride=1, first=False):
        super(ResidualBlock, self).__init__()

        self.net = nn.Sequential(nn.Conv2d(in_channels, channels, 3, padding=1, stride=stride, bias=False),
                                 nn.BatchNorm2d(channels),
                                 nn.ReLU(True),
                                 nn.Conv2d(channels, channels, 3, stride=1, bias=False, padding=1),
                                 nn.BatchNorm2d(channels),
                                 nn.ReLU(True),
                                 nn.Conv2d(channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels)
                                 )

        if first:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                                            nn.BatchNorm2d(out_channels))
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        out = self.net(x)
        out = out + self.downsample(x)
        return F.relu(out)
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, transpose=False):
        super(Up, self).__init__()
        
        if not transpose:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.resblock = ResidualBlock(in_channels+out_channels, out_channels, out_channels, first=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1) #(n-1)s - 2p + k = 2n
            self.resblock = ResidualBlock(in_channels+out_channels, out_channels, out_channels, first=True)
    
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.resblock(x)
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.resblock = ResidualBlock(out_channels, out_channels, out_channels, first=True)
        
    def forward(self, x):
        x = self.down(x)
        return self.resblock(x)

class Generator(nn.Module):
    def __init__(self, input_channel, output_channel, image_size=152, dropout=0.2, transpose=False):
        super().__init__()
        assert image_size % 8 == 0, 'image size must be a multiple of 16'
        
        self.first = ResidualBlock(input_channel, 64, 64, first=True)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        
        self.up1 = Up(512, 256, transpose=transpose)
        self.up2 = Up(256, 128, transpose=transpose)
        self.up3 = Up(128, 64, transpose=transpose)
        
        self.final = nn.Conv2d(64, output_channel, 1)
        self.tanh = nn.Tanh()
        
        self.maxpool = nn.MaxPool2d(2)
        self.dropouts = torch.nn.ModuleList([nn.Dropout2d(dropout) for _ in range(6)])
        
    def forward(self, x): 
        x1 = self.first(x) 
                
        x2 = self.down2(x1) 
        x2 = self.dropouts[0](x2)
        
        x3 = self.down3(x2) 
        x3 = self.dropouts[1](x3)
        
        x4 = self.down4(x3) 
        x4 = self.dropouts[2](x4)
        
        #UPSAMPLING
        u4 = self.up1(x4, x3)
        u4 = self.dropouts[3](u4)
        
        u3 = self.up2(u4, x2)
        u3 = self.dropouts[4](u3)
        
        u2 = self.up3(u3, x1)
        u2 = self.dropouts[5](u2)
        
        return self.tanh(self.final(u2))

        
class Discriminator(nn.Module):
    def __init__(self, image_size=152, in_channels=3, drip=0.2, patch=True):
        super().__init__()
        self.drip=drip
        self.patch = patch
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(drip),
            
            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(drip),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(drip),
            
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(drip),
            nn.Conv2d(512, 1, 1)
        )
        if not patch:
            self.fc = nn.LazyLinear(1)

    def forward(self, *input_data):
        if len(input_data) == 2: #(L, ab)
            l, ab = input_data
            x = torch.cat([l, ab], dim=1).float()
            x = self.conv_layers(x)
        else:
            x = input_data[0]
            x = self.conv_layers(x)

        if not self.patch:
            x = torch.flatten(x, 1)
            return self.fc(x)
        
        return x


def init_weights(m, method='xavier', std=0.02):
    if method == 'xavier':
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
    elif method == 'normal':
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            torch.nn.init.normal_(m.weight, 0.0, std)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
