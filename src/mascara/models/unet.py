
import torch, torch.nn as nn, torch.nn.functional as F
class DoubleConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch,out_ch,3,padding=1,bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Conv2d(out_ch,out_ch,3,padding=1,bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self,x): return self.net(x)
class UNet(nn.Module):
    def __init__(self,in_ch=3,out_ch=1,features=(64,128,256,512)):
        super().__init__(); self.downs = nn.ModuleList(); self.ups = nn.ModuleList(); self.pool = nn.MaxPool2d(2)
        ch = in_ch
        for f in features: self.downs.append(DoubleConv(ch,f)); ch = f
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        for f in reversed(features): self.ups.append(nn.ConvTranspose2d(f*2,f,2,2)); self.ups.append(DoubleConv(f*2,f))
        self.final = nn.Conv2d(features[0], out_ch, 1)
    def forward(self,x):
        skips = []
        for d in self.downs: x = d(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x); skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x); s = skips[i//2]
            if x.shape[2:] != s.shape[2:]: x = F.interpolate(x, size=s.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([s,x], dim=1); x = self.ups[i+1](x)
        return self.final(x)
