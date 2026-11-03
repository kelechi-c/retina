"""retina: an image tokenizer whose token count follows the image's content.

    from retina import Tokenizer, load_img, save_png
    tok = Tokenizer()                      # FLUX.2-dev VAE, tuned DReX budget, S1 refiner
    x = load_img("painting.jpg", 768)      # [1,3,768,768] in [-1,1]
    t = tok.encode(x.cuda())
    print(Tokenizer.length(t))             # tokens this image costs
    save_png(tok.decode(t), "out.png")
    seq = Tokenizer.pack(t)["seq"]         # one [N,128] sequence for an AR model
"""
from .common import load_img, save_png
from .tokenizer import Tokenizer

__all__ = ["Tokenizer", "load_img", "save_png"]
__version__ = "1.0.0"
