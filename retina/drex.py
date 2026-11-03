"""DReX image complexity model (arXiv 2511.16991, github.com/jskaza/DReX): same architecture as its models.py, loaded weights-only."""
import hashlib
import os
import pathlib
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# downloaded once; HF_HOME if set, so it sits with the other weights
CACHE = pathlib.Path(os.environ.get("HF_HOME", pathlib.Path.home() / ".cache" / "huggingface"))
IM = (.485, .456, .406), (.229, .224, .225)


class Fusion(nn.Module):
    def __init__(self, d, r, h=384):
        super().__init__()
        self.dino_proj = nn.Sequential(nn.Linear(d, h), nn.LayerNorm(h), nn.GELU())
        self.resnet_proj = nn.Sequential(nn.Linear(r, h), nn.LayerNorm(h), nn.GELU())
        self.attention = nn.Sequential(nn.Linear(h * 2, 128), nn.Tanh(), nn.Linear(128, 2))
        self.temperature, self.alpha = nn.Parameter(torch.tensor(.7)), nn.Parameter(torch.tensor(.1))

    def forward(self, a, b):
        p, q = self.dino_proj(a), self.resnet_proj(b)
        w = F.softmax(self.attention(torch.cat((p, q), -1)) / self.temperature, -1)
        f = w[:, :1] * p + w[:, 1:2] * q + self.alpha * (p + q)
        return F.layer_norm(f, f.shape[-1:])


def load_drex(dev="cuda"):
    """DReX (arXiv 2511.16991), same architecture as its models.py, weights only, checksum against the repo's LFS pointer. All backbone weights come from the checkpoint."""
    from transformers import DINOv3ViTConfig, DINOv3ViTModel

    p = CACHE / "drex.pth"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        urllib.request.urlretrieve("https://media.githubusercontent.com/media/jskaza/DReX/main/DReX.pth", p)
    assert hashlib.sha256(p.read_bytes()).hexdigest() == "4a7fb1b6c8f2efb845bdf3635ae25336ad875bb8434c32b52129e1b2e560ded2"
    sd = torch.load(p, map_location="cpu", weights_only=True)
    sd = sd.get("model_state", sd)
    dino = DINOv3ViTModel(DINOv3ViTConfig(hidden_size=384, intermediate_size=1536, num_attention_heads=6, num_hidden_layers=12, patch_size=16, num_register_tokens=4)).eval()
    print(dino.load_state_dict({k[11:].replace("layer.", "model.layer.", 1) if k.startswith("dino_model.layer.") else k[11:]: v for k, v in sd.items() if k.startswith("dino_model.")}, strict=True), flush=True)
    r = torchvision.models.resnet50(weights=None).eval()
    print(r.load_state_dict({k[16:]: v for k, v in sd.items() if k.startswith("resnet_backbone.")}, strict=False), flush=True)
    fu = Fusion(384, 256 + 512 + 1024 + 2048).eval()
    fu.load_state_dict({k[7:]: v for k, v in sd.items() if k.startswith("fusion.")})
    h = nn.Sequential(nn.Linear(384, 128), nn.GELU(), nn.Dropout(.1), nn.Linear(128, 64), nn.GELU(), nn.Dropout(.1), nn.Linear(64, 32), nn.GELU(), nn.Dropout(.1), nn.Linear(32, 1)).eval()
    h.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("head.")})
    return [m.to(dev) for m in (dino, r, fu, h)]


@torch.no_grad()
def drex_score(M, x, di=224, rs=512):
    dino, r, fu, h = M
    dev = x.device
    x = (x + 1) / 2
    m, s = torch.tensor(IM[0], device=dev).view(1, 3, 1, 1), torch.tensor(IM[1], device=dev).view(1, 3, 1, 1)
    a = dino(pixel_values=(F.interpolate(x, size=di, mode="bilinear", antialias=True) - m) / s).last_hidden_state[:, 0]
    y = (F.interpolate(x, size=rs, mode="bilinear", antialias=True) - m) / s
    y = r.maxpool(r.relu(r.bn1(r.conv1(y))))
    f = []
    for l in (r.layer1, r.layer2, r.layer3, r.layer4):
        y = l(y)
        f.append(y.mean((2, 3)))
    return h(fu(a, torch.cat(f, -1)))[:, 0].cpu().tolist()
