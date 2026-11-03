"""A small residual net that sits after the tiny decoder and the unwarp. It sees the canvas, how much each pixel was stretched on each axis, and noise, so it can restore texture where stretching removed it and leave native areas alone. Last layer starts at zero: step 0 is the plain codec."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as P


class Res(nn.Module):
    def __init__(self, f, d=1):
        super().__init__()
        self.a, self.b = nn.Conv2d(f, f, 3, padding=d, dilation=d), nn.Conv2d(f, f, 3, padding=1)

    def forward(self, x):
        return x + self.b(F.silu(self.a(F.silu(x))))


class Refiner(nn.Module):
    def __init__(self, f=48, mid=8):
        super().__init__()
        self.head = nn.Conv2d(6, f, 3, padding=1)
        self.down = nn.Conv2d(f, 2 * f, 3, 2, 1)
        self.mid = nn.Sequential(*[Res(2 * f, (1, 2, 4, 8)[i % 4]) for i in range(mid)])
        self.up = nn.Sequential(nn.Conv2d(2 * f, 4 * f, 1), nn.PixelShuffle(2))
        self.fuse = nn.Conv2d(2 * f, f, 1)
        self.tail = nn.Sequential(Res(f), Res(f), Res(f))
        self.out = nn.Conv2d(f, 3, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, c, stretch, noise=None):
        """c [B,3,H,W] in [-1,1]; stretch [B,2,H,W] = log2(source px per coded px) / 4."""
        if noise is None:
            noise = torch.randn_like(c[:, :1])
        h = self.head(torch.cat((c, stretch, noise), 1))
        m = self.up(self.mid(self.down(h)))
        return c + self.out(self.tail(self.fuse(torch.cat((h, m), 1))))


class Disc(nn.Module):
    """Patch discriminator."""

    def __init__(self, f=48):
        super().__init__()
        sn = P.spectral_norm
        self.n = nn.Sequential(sn(nn.Conv2d(3, f, 4, 2, 1)), nn.LeakyReLU(.2, True), sn(nn.Conv2d(f, 2 * f, 4, 2, 1)), nn.LeakyReLU(.2, True), sn(nn.Conv2d(2 * f, 4 * f, 4, 2, 1)), nn.LeakyReLU(.2, True), sn(nn.Conv2d(4 * f, 1, 3, 1, 1)))

    def forward(self, x):
        return self.n(x)


class StatRefiner(Refiner):
    """Refiner that also reads one statistics token per tile, broadcast over that tile. Starts at zero, like the output layer, so step 0 ignores the token."""

    def __init__(self, f=48, mid=8, stats=27):
        super().__init__(f, mid)
        self.stat = nn.Conv2d(stats, f, 1)
        nn.init.zeros_(self.stat.weight)
        nn.init.zeros_(self.stat.bias)

    def forward(self, c, stretch, stats, noise=None):
        if noise is None:
            noise = torch.randn_like(c[:, :1])
        h = self.head(torch.cat((c, stretch, noise), 1)) + self.stat(stats)
        m = self.up(self.mid(self.down(h)))
        return c + self.out(self.tail(self.fuse(torch.cat((h, m), 1))))
