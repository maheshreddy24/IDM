""" Flow matching (Flow-World-Models' transport, 'Linear' path + 'velocity' prediction).

Convention (theirs): t = 0 is data, t = 1 is noise.
    x_t = (1 - t) * x1 + t * eps          x1 = data, eps ~ N(0, I)
    u   = dx_t / dt = eps - x1            regression target for the model's velocity
Sampling integrates dx/dt = v(x, t) from t = 1 (noise) down to t = 0 (data).
"""
import torch


def shift_time(t, shift):
    """ Skews t toward 1 (noisier) for shift > 1: t' = s t / (1 + (s - 1) t). t = 0.5 -> 0.93 for s = 13 """
    return shift * t / (1 + (shift - 1) * t)


def sample_time(batch_size, shift, device):
    """ Training timesteps: uniform, then shifted """
    return shift_time(torch.rand(batch_size, device=device), shift)


def expand_like(t, x):
    return t.view(-1, *([1] * (x.dim() - 1)))


def interpolate(x1, eps, t):
    """ Point on the straight path between data x1 (t = 0) and noise eps (t = 1) """
    t = expand_like(t, x1)
    return (1 - t) * x1 + t * eps


def velocity_target(x1, eps):
    return eps - x1


def predict_x1(xt, t, v):
    """ One-step estimate of the clean data from x_t and the predicted velocity """
    return xt - expand_like(t, xt) * v


@torch.no_grad()
def euler_sample(velocity_fn, noise, num_steps=50, shift=1.0, t_end=1e-3):
    """ Integrate from t = 1 to t ~ 0 with Euler steps on the same shifted time grid used in training,
    then take a final one-step projection to t = 0.

    velocity_fn(x, t) -> v, with t of shape bs; noise: starting point at t = 1 """
    x = noise
    bs = x.shape[0]
    ts = shift_time(1 - torch.linspace(0, 1 - t_end, num_steps, device=x.device), shift)  # 1 -> ~0
    for t_cur, t_next in zip(ts[:-1], ts[1:]):
        v = velocity_fn(x, t_cur.expand(bs))
        x = x + (t_next - t_cur) * v  # t_next < t_cur: steps toward the data
    t_last = ts[-1].expand(bs)
    return predict_x1(x, t_last, velocity_fn(x, t_last))
