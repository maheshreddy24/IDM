""" Flow-matching world model in frozen-backbone feature space.

Adapted from Flow-World-Models (models/DDT.py: DiTwDDTHead and the helpers it uses from models/model_utils.py),
reduced to the configuration used there (RMSNorm, QK-norm, SwiGLU, learnable sin-cos positions) and to one
context frame and one target frame, so features are handled as token sequences bs, N, C.

It predicts the flow-matching velocity for the noised target features, conditioned on (timestep t, latent action a)
through AdaLN (their "adaln" action mode) and on the context features through cross-attention in every block.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return out.type_as(x) * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class Attention(nn.Module):
    """ Multi-head attention with QK-norm; self-attention when context is None, cross-attention otherwise """
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, context=None):
        """ x: B, N, C; context: B, M, C (defaults to x) -> B, N, C """
        context = x if context is None else context
        B, N, C = x.shape
        M = context.shape[1]
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        # B, M, 2 C -> 2, B, num_heads, M, head_dim
        k, v = self.kv(context).reshape(B, M, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


def modulate(x, shift, scale):
    """ x: B, N, C; shift, scale: B, 1, C or B, N, C """
    return x * (1 + scale) + shift


class DDTBlock(nn.Module):
    """ AdaLN-Zero block: self-attn over target tokens, cross-attn to context tokens, FFN.
    Each sub-layer is modulated (shift, scale) and gated by the conditioning c """
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1, self.norm2, self.norm3, self.norm_context = RMSNorm(dim), RMSNorm(dim), RMSNorm(dim), RMSNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.cross_attn = Attention(dim, num_heads)
        self.mlp = SwiGLUFFN(dim, int(2 / 3 * dim * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))

    def forward(self, x, c, context):
        """ x: B, N, C target tokens; c: B, C or B, N, C conditioning; context: B, M, C """
        if c.dim() == 2:
            c = c.unsqueeze(1)  # B, 1, C
        (shift_msa, scale_msa, gate_msa,
         shift_ca, scale_ca, gate_ca,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(9, dim=-1)

        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_ca * self.cross_attn(modulate(self.norm2(x), shift_ca, scale_ca), self.norm_context(context))
        x = x + gate_mlp * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.linear = nn.Linear(dim, out_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))  # no gate


class GaussianFourierEmbedding(nn.Module):
    """ Timestep embedding: fixed random Fourier features -> MLP """
    def __init__(self, dim, embedding_size=256, scale=1.0):
        super().__init__()
        self.register_buffer('W', torch.randn(embedding_size) * scale)
        self.mlp = nn.Sequential(nn.Linear(embedding_size * 2, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        """ t: B -> B, dim """
        t = t[:, None] * self.W[None, :] * 2 * torch.pi
        return self.mlp(torch.cat([torch.sin(t), torch.cos(t)], dim=-1))


def sincos_pos_embed(dim, h, w):
    """ h * w, dim fixed sin-cos table: h and w get (dim // 6) * 2 channels each, the rest are the (single) time
    step's channels, as in Flow-World-Models' PositionalEncoding3D with T = 1 """
    dim_hw = (dim // 6) * 2
    dim_t = dim - 2 * dim_hw

    def sincos_1d(n, d):
        pos = torch.arange(n, dtype=torch.float32)
        omega = 1.0 / (10000 ** (2 * torch.arange(d // 2, dtype=torch.float32) / d))
        out = pos[:, None] * omega[None, :]
        return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)

    grid_h, grid_w = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    return torch.cat([
        sincos_1d(1, dim_t).expand(h * w, -1),
        sincos_1d(h, dim_hw)[grid_h.flatten()],
        sincos_1d(w, dim_hw)[grid_w.flatten()],
    ], dim=-1)


class WorldModel(nn.Module):
    """ DiT with a wide DDT head, predicting the flow-matching velocity of the next frame's features.

    Two stages:
      - encoder (width hidden_size[0], depth[0] blocks): embeds the noisy target x and the context, runs blocks
        conditioned on c = silu(t_emb) + a_emb that cross-attend to the context -> s, a per-token condition.
      - decoder (width hidden_size[1], depth[1] blocks): re-embeds x and the context at the wider width and runs
        blocks conditioned per token on s + proj(a_emb), again cross-attending to the context.
    The final layer (AdaLN on s) maps back to in_dim.

    x (noisy target), context: bs, N, in_dim; t: bs; action: bs, 1, action_dim -> velocity bs, N, in_dim
    """
    def __init__(self, in_dim=1024, action_dim=256, grid_size=(16, 16), hidden_size=(256, 1024), depth=(2, 2),
                 num_heads=(8, 16), mlp_ratio=4.0):
        super().__init__()
        enc_dim, dec_dim = hidden_size
        self.in_dim = in_dim

        self.t_embedder = GaussianFourierEmbedding(enc_dim)
        self.action_embedder = nn.Sequential(nn.Linear(action_dim, enc_dim), nn.SiLU(), nn.Linear(enc_dim, enc_dim))
        self.action_to_decoder = nn.Linear(enc_dim, dec_dim)

        # encoder stage
        self.s_embedder = nn.Linear(in_dim, enc_dim)
        self.s_context_embedder = nn.Linear(in_dim, enc_dim)
        self.encoder_blocks = nn.ModuleList([DDTBlock(enc_dim, num_heads[0], mlp_ratio) for _ in range(depth[0])])
        self.s_projector = nn.Linear(enc_dim, dec_dim)
        # decoder stage
        self.x_embedder = nn.Linear(in_dim, dec_dim)
        self.x_context_embedder = nn.Linear(in_dim, dec_dim)
        self.decoder_blocks = nn.ModuleList([DDTBlock(dec_dim, num_heads[1], mlp_ratio) for _ in range(depth[1])])
        self.final_layer = FinalLayer(dec_dim, in_dim)

        # learnable positions, sin-cos init; separate tables for target and context tokens, per stage
        h, w = grid_size
        self.pos_s_target = nn.Parameter(sincos_pos_embed(enc_dim, h, w))
        self.pos_s_context = nn.Parameter(sincos_pos_embed(enc_dim, h, w))
        self.pos_x_target = nn.Parameter(sincos_pos_embed(dec_dim, h, w))
        self.pos_x_context = nn.Parameter(sincos_pos_embed(dec_dim, h, w))

        self.initialize_weights()

    def initialize_weights(self):
        for embedder in (self.s_embedder, self.s_context_embedder, self.x_embedder, self.x_context_embedder, self.s_projector):
            nn.init.xavier_uniform_(embedder.weight)
            nn.init.zeros_(embedder.bias)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        # AdaLN-Zero: every block starts as the identity, and the initial velocity prediction is 0
        for block in (*self.encoder_blocks, *self.decoder_blocks):
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        for layer in (self.final_layer.adaLN_modulation[-1], self.final_layer.linear):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, t, context, action):
        t = self.t_embedder(t)  # bs, enc_dim
        action = self.action_embedder(action.reshape(action.shape[0], -1))  # bs, 1, action_dim -> bs, enc_dim
        c = F.silu(t) + action  # (time, action) conditioning for the encoder stage

        # encoder stage
        s = self.s_embedder(x) + self.pos_s_target
        s_context = self.s_context_embedder(context) + self.pos_s_context
        for block in self.encoder_blocks:
            s = block(s, c, s_context)

        # per-token condition for the decoder stage
        s = self.s_projector(F.silu(t.unsqueeze(1) + s)) + self.action_to_decoder(action).unsqueeze(1)  # bs, N, dec_dim

        # decoder stage
        x = self.x_embedder(x) + self.pos_x_target
        x_context = self.x_context_embedder(context) + self.pos_x_context
        for block in self.decoder_blocks:
            x = block(x, s, x_context)
        return self.final_layer(x, s)
