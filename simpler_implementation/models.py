import torch
import torch.nn as nn
import torch.nn.functional as F
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, dropout=False):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1) #(n-1)s - 2p + k = 2n
        self.dropout = dropout
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        if dropout:
            self.drop = nn.Dropout2d(0.5)
        self.relu = nn.LeakyReLU(0.2, True)
    
    def forward(self, x, skip=None):
        x = self.up(self.relu(x))
        if self.batch_norm:
            x = self.bn(x)
        if self.dropout:
            x = self.drop(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return x
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, activation=True, batch_norm=True):
        super(Down, self).__init__()
        self.down = nn.Conv2d(in_channels, out_channels, 4, stride=stride, padding=1)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        self.batch_norm = batch_norm
        if activation:
            self.leaky_relu = nn.LeakyReLU(0.2)
        self.activation = activation
        
    def forward(self, x):
        x = self.down(x)
        if self.activation:
            x = self.leaky_relu(x)
        if self.batch_norm:
            x = self.bn(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, image_size=256):
        super().__init__()
        assert image_size % 2**8 == 0, 'image size must be a multiple of 256'
        
        self.down1 = Down(in_channels, 64, activation=False, batch_norm=False)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 512)
        self.down6 = Down(512, 512)
        self.down7 = Down(512, 512)
        self.down8 = Down(512, 512, batch_norm = False)
        
        self.up1 = Up(512, 512, dropout=True)
        self.up2 = Up(512*2, 512, dropout=True)
        self.up3 = Up(512*2, 512, dropout=True)
        self.up4 = Up(512*2, 512)
        self.up5 = Up(512*2, 256)
        self.up6 = Up(256*2, 128)
        self.up7 = Up(128*2, 64)
        self.up8 = Up(64*2, out_channels, batch_norm=False)
        
        self.tanh = nn.Tanh()
        
    def forward(self, x):
         
        #DOWNSAMPLING
        x1 = self.down1(x) 
        x2 = self.down2(x1) 
        x3 = self.down3(x2) 
        x4 = self.down4(x3) 
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x7 = self.down7(x6)
        x8 = self.down8(x7) 
        
        #UPSAMPLING
        u1 = self.up1(x8, x7)
        u2 = self.up2(u1, x6)
        u3 = self.up3(u2, x5)
        u4 = self.up4(u3, x4)
        u5 = self.up5(u4, x3)
        u6 = self.up6(u5, x2)
        u7 = self.up7(u6, x1)
        u8 = self.up8(u7)

        
        return self.tanh(u8)

class Discriminator(nn.Module):
    def __init__(self, in_channels, patch=True):
        super().__init__()
        self.conv1 = Down(in_channels, 64, activation=False, batch_norm=False)
        self.conv2 = Down(64, 128)
        self.conv3 = Down(128, 256)
        self.conv4 = Down(256, 512, stride=1)
        self.conv5 = Down(512, 1, stride=1, batch_norm=False, activation=False)
        
        self.patch = patch
        if not patch:
            self.fc = nn.Linear(1*30*30, 1)
        
    def forward(self, l, ab):
        x = torch.cat([l, ab], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        if not self.patch:
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x

def init_weights(net, std=0.02):
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, std)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, std)
            nn.init.constant_(m.bias.data, 0.0)
    print(f"Weights of {net.__class__.__name__} initialized using normal distribution with std:", std)