import math
import torch
import torch.nn as nn
from .transformer_block import Transformer


def sincos_1d(dim, length):
    """ length, dim sinusoidal table (sin on even channels, cos on odd) """
    pos = torch.arange(length, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
    pe = torch.zeros(length, dim)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def sincos_3d(dim, t, h, w):
    """ t * h * w, dim fixed position embedding, as in LaWAM's Fixed3DPositionalEncoding.
    t gets dim // 2 channels, h dim // 4, w the rest. Each axis is zero-padded at the end to
    dim and the three are summed, so the axes overlap on the leading channels (LaWAM does the same) """
    t_dim, h_dim = dim // 2, dim // 4
    w_dim = dim - t_dim - h_dim
    pe_t = nn.functional.pad(sincos_1d(t_dim, t), (0, dim - t_dim))[:, None, None]  # t, 1, 1, dim
    pe_h = nn.functional.pad(sincos_1d(h_dim, h), (0, dim - h_dim))[None, :, None]  # 1, h, 1, dim
    pe_w = nn.functional.pad(sincos_1d(w_dim, w), (0, dim - w_dim))[None, None, :]  # 1, 1, w, dim
    return (pe_t + pe_h + pe_w).reshape(t * h * w, dim)


def modal_mask(num_patches, num_queries):
    """ bool N, N with N = num_patches + num_queries (True = may attend).
    Patches attend to all patches (both frames) but not to the queries; queries attend to everything """
    is_query = torch.arange(num_patches + num_queries) >= num_patches
    same = is_query[:, None] == is_query[None, :]
    return same | is_query[:, None]


class CrossAttnBlock(nn.Module):
    """ Queries read from the context once before the self-attention stack (LaWAM's CrossAttentionBlock) """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, q, kv):
        """ q: bs, Q, dim; kv: bs, N, dim -> bs, Q, dim """
        kv = self.norm1(kv)
        q = q + self.attn(q, kv, kv)[0]
        q = self.norm2(q)
        return q + self.ffn(q)


class VAEBottleneck(nn.Module):
    """ Continuous bottleneck on the latent action: LN -> (mu, logvar), sample z = mu + sigma * eps while
    training, z = mu at eval. Returns the KL to N(0, I); weight it (LaWAM: beta = 5e-5) in the loss.
    With action_dim set, mu / logvar are down-projected to action_dim (a hard cap on how much the action can carry)
    and z is projected back up to dim for the world model; action_dim=None keeps them at dim (no projection) """
    def __init__(self, dim, action_dim=None, clamp_logvar=10.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mu = nn.Linear(dim, action_dim or dim)
        self.logvar = nn.Linear(dim, action_dim or dim)
        self.up = nn.Linear(action_dim, dim) if action_dim else nn.Identity()
        self.clamp_logvar = clamp_logvar

    def forward(self, x):
        """ x: bs, Q, dim -> dict with z (sampled while training) and z_mean: bs, 1, dim, the world model's input
        (queries are averaged); mu, logvar: bs, 1, action_dim; kl: scalar; kl_per_dim: action_dim (batch mean) """
        h = self.norm(x.mean(dim=1, keepdim=True))
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-self.clamp_logvar, self.clamp_logvar)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if self.training else mu
        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).mean(dim=(0, 1))
        return {'z': self.up(z), 'z_mean': self.up(mu), 'mu': mu, 'logvar': logvar,
                'kl': kl_per_dim.sum(), 'kl_per_dim': kl_per_dim}


class IDM(nn.Module):
    """ Inverse dynamics model: given frozen-backbone features of (x_{n-1}, x_n), infer the latent action a_n.

    features (bs, 2, N, input_dim) -> proj -> + 3D sin-cos pos -> flatten (bs, 2N, emb_dim)
    learned queries cross-attend to the 2N patch tokens once, are appended, then `num_layers` masked
    self-attention blocks run over [patches, queries]; the query outputs -> out_proj -> VAE -> z (bs, 1, latent_dim)
    (through an action_dim-wide code if action_dim is set)
    """
    def __init__(self, input_dim=1024, emb_dim=1024, num_layers=24, num_heads=16, ffn_ratio=4,
                 latent_dim=256, grid_size=(16, 16), num_queries=1, action_dim=None):
        super().__init__()
        gh, gw = grid_size
        num_patches = 2 * gh * gw

        self.proj = nn.Linear(input_dim, emb_dim)
        self.register_buffer('pos_embed', sincos_3d(emb_dim, 2, gh, gw), persistent=False)
        self.queries = nn.Parameter(torch.randn(1, num_queries, emb_dim))
        self.query_cross_attn = CrossAttnBlock(emb_dim, num_heads)
        self.model = Transformer(dim=emb_dim, layers=num_layers, num_heads=num_heads, mlp_ratio=ffn_ratio)
        self.register_buffer('mask', modal_mask(num_patches, num_queries), persistent=False)

        self.out_proj = nn.Linear(emb_dim, latent_dim)
        self.bottleneck = VAEBottleneck(latent_dim, action_dim)

    def forward(self, features):
        """ features: bs, 2, N, input_dim -> the bottleneck's dict (z, z_mean: bs, 1, latent_dim; mu, logvar, kl, kl_per_dim) """
        bs = features.shape[0]
        # bs, 2, N, input_dim -> bs, 2N, emb_dim
        tokens = self.proj(features).flatten(1, 2) + self.pos_embed

        queries = self.query_cross_attn(self.queries.expand(bs, -1, -1), tokens)  # bs, Q, emb_dim
        num_queries = queries.shape[1]
        out = self.model(torch.cat([tokens, queries], dim=1), mask=self.mask)  # bs, 2N + Q, emb_dim

        return self.bottleneck(self.out_proj(out[:, -num_queries:]))
