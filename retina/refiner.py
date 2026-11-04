"""S1, the refiner: a small residual net after the VAE decoder and the unwarp. It sees the decode, how much each pixel was stretched on each axis, one statistics token per 256 tile and noise, so it can restore texture where shrinking removed it and leave native areas alone. Its output layer and its statistics layer start at zero, so step 0 is the plain decode."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Res(nn.Module):
    def __init__(self, f, d=1):
        super().__init__()
        self.a, self.b = nn.Conv2d(f, f, 3, padding=d, dilation=d), nn.Conv2d(f, f, 3, padding=1)

    def forward(self, x):
        return x + self.b(F.silu(self.a(F.silu(x))))


class StatRefiner(nn.Module):
    def __init__(self, f=48, mid=8, stats=27):
        super().__init__()
        self.head = nn.Conv2d(6, f, 3, padding=1)
        self.down = nn.Conv2d(f, 2 * f, 3, 2, 1)
        self.mid = nn.Sequential(*[Res(2 * f, (1, 2, 4, 8)[i % 4]) for i in range(mid)])
        self.up = nn.Sequential(nn.Conv2d(2 * f, 4 * f, 1), nn.PixelShuffle(2))
        self.fuse = nn.Conv2d(2 * f, f, 1)
        self.tail = nn.Sequential(Res(f), Res(f), Res(f))
        self.out = nn.Conv2d(f, 3, 3, padding=1)
        self.stat = nn.Conv2d(stats, f, 1)
        for m in (self.out, self.stat):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, c, stretch, stats, noise=None):
        """c [B,3,H,W] in [-1,1]; stretch [B,2,H,W] = log2(source px per coded px) / 4; stats [B,27,H,W], each tile's token spread over the tile."""
        if noise is None:
            noise = torch.randn_like(c[:, :1])
        h = self.head(torch.cat((c, stretch, noise), 1)) + self.stat(stats)
        m = self.up(self.mid(self.down(h)))
        return c + self.out(self.tail(self.fuse(torch.cat((h, m), 1))))
