""" IDM + flow-matching world model, trained jointly.

    IDM(f_{n-1}, f_n)                       -> a_n  (latent action)
    WorldModel(x_t, t, context=f_{n-1}, a_n) -> v    (flow-matching velocity toward f_n)

All features are frozen-backbone patch features bs, N, C, standardised per channel with statistics estimated
from the training data (see `set_feature_stats`), so they are on the same scale as the N(0, I) noise.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import flow
from .idm import IDM
from .world_model import WorldModel


class LatentActionWorldModel(nn.Module):
    def __init__(self, idm: IDM, world_model: WorldModel, time_shift=3.0, kl_weight=5e-5):
        super().__init__()
        self.idm = idm
        self.world_model = world_model
        self.time_shift = time_shift
        self.kl_weight = kl_weight
        self.register_buffer('feat_mean', torch.zeros(world_model.in_dim))
        self.register_buffer('feat_std', torch.ones(world_model.in_dim))

    @torch.no_grad()
    def set_feature_stats(self, mean, std):
        self.feat_mean.copy_(mean)
        self.feat_std.copy_(std.clamp_min(1e-4))

    def normalize(self, feats):
        return (feats - self.feat_mean) / self.feat_std

    def forward(self, pair):
        """ pair: bs, 2, N, C raw backbone features of (x_{n-1}, x_n).
        The IDM sees both frames; the world model gets frame 0 as context and frame 1 as target.
        -> dict of losses; 'loss' is the total """
        pair = self.normalize(pair)
        action = self.idm(pair)  # z: bs, 1, latent_dim
        context, x1 = pair[:, 0], pair[:, 1]

        t = flow.sample_time(x1.shape[0], self.time_shift, x1.device)
        eps = torch.randn_like(x1)
        v = self.world_model(flow.interpolate(x1, eps, t), t, context, action['z'])

        loss_flow = F.mse_loss(v.float(), flow.velocity_target(x1, eps).float())
        return {
            'loss': loss_flow + self.kl_weight * action['kl'],
            'loss_flow': loss_flow.detach(),
            'loss_kl': action['kl'].detach(),
            'action_std': action['mu'].detach().float().std(dim=0).mean(),  # collapse check: ~0 means one action for everything
        }

    @torch.no_grad()
    def infer_action(self, pair):
        """ bs, 2, N, C raw features -> a_n (mu), bs, 1, latent_dim """
        return self.idm(self.normalize(pair))['mu']

    @torch.no_grad()
    def predict(self, context, action, num_steps=50):
        """ Sample f_n given f_{n-1} and a_n. context: bs, N, C raw features; action: bs, 1, latent_dim -> bs, N, C raw """
        ctx = self.normalize(context)
        x1 = flow.euler_sample(lambda x, t: self.world_model(x, t, ctx, action),
                               torch.randn_like(ctx), num_steps=num_steps, shift=self.time_shift)
        return x1 * self.feat_std + self.feat_mean
