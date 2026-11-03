"""Elastic grid: resample an image onto a smaller rectangle with content-dependent spacing, separably per axis, so flat rows and columns shrink and detail stays at native scale.

A warp is two monotone knot lists (source coordinate at evenly spaced output coordinates). Forward uses exact area averaging from a summed table, so a native-rate stretch is the identity. Inverse is bicubic lookup."""
import torch
import torch.nn.functional as F


def energy(x, axis, p=4., sigma=1.):
    """Importance per row (axis=2) or column (axis=3): edge strength along that axis, pooled over the other axis by a p-norm so a short line still counts."""
    k = torch.exp(-torch.arange(-3, 4, device=x.device).float() ** 2 / (2 * sigma ** 2))
    k = (k / k.sum())
    s = F.conv2d(x, k.view(1, 1, 1, -1).repeat(3, 1, 1, 1), padding=(0, 3), groups=3)
    s = F.conv2d(s, k.view(1, 1, -1, 1).repeat(3, 1, 1, 1), padding=(3, 0), groups=3)
    d = (s.diff(dim=axis).abs().sum(1, keepdim=True))
    other = 3 if axis == 2 else 2
    e = (d ** p).mean(other).pow(1 / p).flatten(1)
    return torch.cat((e[0], e[0, -1:]))


def _spread(e, width):
    k = torch.ones(2 * width + 1, device=e.device)
    m = F.max_pool1d(e[None, None], 2 * width + 1, 1, width)[0, 0]  # keep the surroundings of an edge
    return F.conv1d(m[None, None], k.view(1, 1, -1) / k.sum(), padding=width)[0, 0]


def density_at(e, t, width=4, rmin=.04):
    return (_spread(e, width) / t).clamp(rmin, 1)


def density(e, n, width=4, rmin=.04):
    """Water-filling: rho in [rmin, 1], rising with importance, integrating to n."""
    m = _spread(e, width)
    lo, hi = 1e-6, float(m.max()) * 1e3 + 1e-3
    for _ in range(60):
        t = (lo * hi) ** .5
        s = (m / t).clamp(rmin, 1).sum().item()
        lo, hi = (t, hi) if s > n else (lo, t)
    return (m / (lo * hi) ** .5).clamp(rmin, 1)


def knots(rho, n, K=64):
    """Source coordinates in [0,L] of K+1 evenly spaced output coordinates in [0,n]."""
    c = torch.cat((torch.zeros(1, device=rho.device), rho.cumsum(0)))
    c = c / c[-1] * n
    u = torch.linspace(0, n, K + 1, device=rho.device)
    i = torch.searchsorted(c, u).clamp(1, len(c) - 1)
    a = (u - c[i - 1]) / (c[i] - c[i - 1]).clamp_min(1e-9)
    return i - 1 + a


def _table(kn, n):
    """Knots -> source coordinate at each integer output boundary 0..n."""
    u = torch.arange(n + 1, device=kn.device).float() * (len(kn) - 1) / n
    i = u.long().clamp(0, len(kn) - 2)
    return kn[i] + (kn[i + 1] - kn[i]) * (u - i)


def _area(x, Y, dim):
    """Area average of x between consecutive source coordinates Y along dim."""
    x = x.movedim(dim, -1)
    c = torch.cat((torch.zeros_like(x[..., :1]), x.cumsum(-1)), -1)
    i = Y.long().clamp(0, x.shape[-1] - 1)
    a = (Y - i).view(*[1] * (x.ndim - 1), -1)
    v = c[..., i] * (1 - a) + c[..., i + 1] * a
    return (v.diff(dim=-1) / Y.diff().clamp_min(1e-6).view(*[1] * (x.ndim - 1), -1)).movedim(-1, dim)


def forward(x, ky, kx, ny, nx):
    return _area(_area(x, _table(ky, ny), 2), _table(kx, nx), 3)


def _inverse_table(kn, n, L):
    """For each of L source pixel centres, the output coordinate in [0,n]."""
    Y = _table(kn, n)
    s = torch.arange(L, device=kn.device).float() + .5
    i = torch.searchsorted(Y, s).clamp(1, n)
    a = (s - Y[i - 1]) / (Y[i] - Y[i - 1]).clamp_min(1e-6)
    return i - 1 + a


def inverse(y, ky, kx, H, W, mode="bicubic"):
    """Warped image y -> [1,3,H,W]."""
    ny, nx = y.shape[-2:]
    v = _inverse_table(ky, ny, H) / ny * 2 - 1
    u = _inverse_table(kx, nx, W) / nx * 2 - 1
    g = torch.stack(torch.meshgrid(v, u, indexing="ij")[::-1], -1)[None]
    return F.grid_sample(y, g, mode=mode, padding_mode="border", align_corners=False).clamp(-1, 1)


def local_scale(ky, kx, ny, nx, H, W):
    """Stretch factor per axis at each source pixel (1 = native, larger = more stretched), [1,2,H,W]."""
    def one(k, n, L):
        Y = _table(k, n)
        s = Y.diff()
        return s[(_inverse_table(k, n, L).long().clamp(0, n - 1))]

    sy, sx = one(ky, ny, H), one(kx, nx, W)
    return torch.stack((sy[:, None].expand(H, W), sx[None].expand(H, W)))[None]
