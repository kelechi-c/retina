"""The VAEs behind the tokenizer: FLUX.2-dev (default), FLUX.1-dev and Qwen-Image 2.1, each as encode and raw decode."""
import torch
from diffusers import AutoencoderKL


class Flux1:
    z = 16

    def __init__(self, dev):
        self.dev = dev
        self.vae = self.load().to(dev).eval().requires_grad_(False)
        self.a, self.b = torch.ones(self.z, device=dev), torch.zeros(self.z, device=dev)

    def load(self):
        return AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-schnell", subfolder="vae", torch_dtype=torch.float32)

    @torch.no_grad()
    def encode(self, x):
        return self.vae.encode(x).latent_dist.mode()

    @torch.no_grad()
    def fit(self, cod, xs):
        """xs: [1,3,S,S] images. Least squares zf = a zt + b per channel."""
        zt, zf = [], []
        for x in xs:
            zt.append(cod.enc(x).transpose(0, 1).flatten(1))
            zf.append(self.encode(x).transpose(0, 1).flatten(1))
        zt, zf = torch.cat(zt, 1), torch.cat(zf, 1)
        mt, mf = zt.mean(1), zf.mean(1)
        self.a = ((zt.T - mt) * (zf.T - mf)).sum(0) / (zt.T - mt).pow(2).sum(0)
        self.b = mf - self.a * mt
        r = torch.stack([torch.corrcoef(torch.stack((zt[c], zf[c])))[0, 1] for c in range(self.z)])
        return self.a.cpu().tolist(), self.b.cpu().tolist(), r.cpu().tolist()

    @torch.no_grad()
    def dec_raw(self, z):
        return self.vae.decode(z).sample.clamp(-1, 1)

    @torch.no_grad()
    def dec(self, z):
        return self.dec_raw(z * self.a.view(1, -1, 1, 1) + self.b.view(1, -1, 1, 1))


class Flux2(Flux1):
    """FLUX.2 VAE, 32 channels at f8; pairs with TAEF2, not TAEF1."""

    z = 32
    repo = "black-forest-labs/FLUX.2-dev"

    def load(self):
        from diffusers import AutoencoderKLFlux2

        return AutoencoderKLFlux2.from_pretrained(self.repo, subfolder="vae", torch_dtype=torch.float32)


class Qwen:
    """Qwen-Image 2.1 VAE, f16 with 64 channels, so one latent cell is one 64-float token. RGB in and out, one image as one frame; alpha is dropped."""

    def __init__(self, dev):
        from diffusers import AutoencoderKLQwenImage21

        self.dev = dev
        self.vae = AutoencoderKLQwenImage21.from_pretrained("Qwen/Qwen-Image-2.1-Turbo", subfolder="vae", torch_dtype=torch.bfloat16).eval().to(dev).requires_grad_(False)

    @torch.no_grad()
    def enc(self, x):
        rgba = torch.cat((x, torch.ones_like(x[:, :1])), 1)[:, :, None].to(self.dev, torch.bfloat16)
        return self.vae.encode(rgba).latent_dist.mode()[:, :, 0]

    @torch.no_grad()
    def dec(self, z):
        return self.vae.decode(z[:, :, None].to(self.dev, torch.bfloat16)).sample[:, :, 0].float()[:, :3].clamp(-1, 1)
