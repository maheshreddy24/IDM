import torch
import torch.nn as nn
from transformers import AutoModel


class FFN(nn.Module):
    def __init__(self, dim, ratio):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * ratio),
            nn.GELU(),
            nn.Linear(dim * ratio, dim)
        )

    def forward(self, x):
        return self.mlp(x)


class SelfAttn(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.output = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        """ x.shape: bs, N, dim; mask: optional bool N, N (True = may attend) """
        bs, n, _ = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)
        # bs, n, num_heads * head_dim -> bs, num_heads, n, head_dim
        q, k, v = (t.view(bs, n, self.num_heads, self.head_dim).transpose(1, 2) for t in (q, k, v))

        # bs, num_heads, n, n
        attn = (q @ k.transpose(-2, -1)) * self.head_dim ** -0.5
        if mask is not None:
            attn = attn.masked_fill(~mask, float('-inf'))
        attn = attn.softmax(dim=-1)

        # bs, num_heads, n, head_dim -> bs, n, num_heads * head_dim
        out = (attn @ v).transpose(1, 2).reshape(bs, n, self.num_heads * self.head_dim)
        return self.output(out)


class Block(nn.Module):
    """ Pre-norm transformer block: x + attn(norm(x)), then x + ffn(norm(x)) """
    def __init__(self, dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttn(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FFN(dim, mlp_ratio)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class Transformer(nn.Module):
    """ Stack of `layers` blocks followed by a final LayerNorm """
    def __init__(self, dim, layers, num_heads, mlp_ratio=4):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, num_heads, mlp_ratio) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)


class Encoder(nn.Module):
    """ Frozen DINOv2 / DINOv3 backbone. Expects ImageNet-normalised images.

    Features follow Flow-World-Models: the outputs of `layers` each go through the backbone's final norm and are
    averaged (they use layers 3, 6, 9, 12 of the 12-layer ViT-S; the default is the same spacing over the
    24-layer ViT-L). Pass a single layer, e.g. layers=(24,), to use one layer only """
    def __init__(self, model_name='facebook/dinov2-large', layers=(6, 12, 18, 24)):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.patch_size = self.model.config.patch_size
        self.dim = self.model.config.hidden_size
        # CLS + register tokens come before the patch tokens
        self.norm = getattr(self.model, 'norm', None) or self.model.layernorm  # DINOv3: norm, DINOv2: layernorm
        self.num_prefix_tokens = 1 + getattr(self.model.config, 'num_register_tokens', 0)
        self.layers = layers

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        # keep the backbone in eval mode even when the parent module is set to train
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, x):
        """ x.shape: bs, 3, H, W -> bs, num_patches, dim (CLS and register tokens dropped) """
        # hidden_states[0] is the patch embedding, hidden_states[i] the output of block i
        hidden_states = self.model(pixel_values=x, output_hidden_states=True).hidden_states
        feats = [self.norm(hidden_states[i][:, self.num_prefix_tokens:]) for i in self.layers]
        return torch.stack(feats, dim=0).mean(dim=0)
