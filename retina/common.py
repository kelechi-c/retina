"""Shared image helpers. Images are tensors [1,3,S,S] in [-1,1]."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps


def down(x, side):
    return F.interpolate(x, size=side, mode="bicubic", antialias=True, align_corners=False)


def up(x, side):
    return F.interpolate(x, size=side, mode="bicubic", align_corners=False).clamp(-1, 1)


def to8(x):
    """Rounded to 8 bit, as a saved image would be."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round() / 127.5 - 1


def save_png(x, path):
    a = ((x[0].clamp(-1, 1) + 1) * 127.5).round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(a).save(path)


def load_img(path, side):
    """Centre crop to a square, resize to side."""
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")  # phone photos store their rotation in EXIF
    s = min(im.size)
    l, u = (im.width - s) // 2, (im.height - s) // 2
    a = np.asarray(im.crop((l, u, l + s, u + s)).resize((side, side), Image.LANCZOS)).copy()
    return torch.from_numpy(a).permute(2, 0, 1).float()[None] / 127.5 - 1
