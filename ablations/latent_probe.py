""" Ablation: what does the latent action a_t = IDM(s_{t-1}, s_t) encode about the ball (parabola data)?

Linear probes (ridge, LOO-selected strength) on the IDM's bottleneck code mu (action_dim wide; the world model gets
a linear map of it), fitted on a growing number of training pairs and evaluated on held-out pairs:
    a. position   ball centre at t in pixels                  error (x - x^)^2 + (y - y^)^2        [px^2]
                  and at t-1 (the context frame): p_t decoding better than p_{t-1} -> a_t stores the future location
    b. velocity   displacement p_t - p_{t-1}, x and y         error (v - v^)^2 per direction       [px^2]
    c. diameter   2 * radius, constant per clip               error (d - d^)^2                     [px^2]
Pixels are those of the 256 x 256 video (the 10 x 10 world fills it; y points down). Only pairs with the ball's
centre inside the frame at t-1 and t are used. Chance: predict the mean of the training pool.
Each size is refitted on --repeats random subsets of the training pool (mean +- std in the plot).

    python ablations/latent_probe.py runs/parabola/step_10000.pt [--data EVAL.hdf5] [--sizes 100 300 ... 30000]
        [--repeats 5] [--device cuda]

Writes to <checkpoint dir>/ablations/latent_probe_<checkpoint name>[_<data name>]/: results.json, probe_scaling.png
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import VideoPairDataset, list_clips
from src.utils import fit_ridge, ridge_predict
from train import build_models, encode_pairs

PX = 25.6  # pixels per world unit
FRAME = 256  # video frame size in pixels


@torch.no_grad()
def load(model, encoder, dataset, idx, device, bs, num_workers):
    """ -> mu N, action_dim (float32), positions N, 2, 2 (t-1 / t, xy world units), init N, 2 (radius, vx world units) """
    mus, positions, inits = [], [], []
    for i, batch in enumerate(DataLoader(Subset(dataset, idx), batch_size=bs, num_workers=num_workers)):
        if 'init' not in batch:
            raise ValueError(f'{dataset.path} has no init_streams (parabola data only)')
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.startswith('cuda')):
            pair = encode_pairs(encoder, batch['frames'].to(device))
            mus.append(model.idm(model.normalize(pair))['mu'].squeeze(1).float().cpu())
        positions.append(batch['positions'])
        inits.append(batch['init'])
        if (i + 1) % 50 == 0:
            print(f"  {(i + 1) * bs} / {len(idx)} pairs", flush=True)
    return torch.cat(mus), torch.cat(positions), torch.cat(inits)


def targets(positions, init):
    """ -> {task: N, K} in pixels, visible N bool (ball centre inside the frame at t-1 and t) """
    px = torch.stack([positions[..., 0] * PX, FRAME - positions[..., 1] * PX], dim=-1)  # N, 2, 2, y down
    visible = ((px >= 0) & (px < FRAME)).flatten(1).all(1)
    return {'position': px[:, 1], 'position_prev': px[:, 0], 'velocity': px[:, 1] - px[:, 0], 'diameter': 2 * PX * init[:, :1]}, visible


def errors(pred, y):
    """ task predictions -> {curve: scalar} mean squared errors """
    se = {k: (pred[k] - y[k]).pow(2) for k in y}
    return {'position': se['position'].sum(-1).mean().item(), 'position_prev': se['position_prev'].sum(-1).mean().item(),
            'velocity_x': se['velocity'][:, 0].mean().item(),
            'velocity_y': se['velocity'][:, 1].mean().item(), 'diameter': se['diameter'].mean().item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--sizes', type=int, nargs='+', default=[30, 100, 300, 1000, 3000, 10000, 30000],
                        help='numbers of training pairs the probes are fitted on')
    parser.add_argument('--repeats', type=int, default=5, help='random training subsets per size')
    parser.add_argument('--num', type=int, default=2000, help='held-out clips (one pair each)')
    parser.add_argument('--data', default=None, help='hdf5 to evaluate on, all clips (default: val split of the training file)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    gen = torch.Generator().manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg, data_cfg = ckpt['config'], ckpt['config']['data']
    encoder, model = build_models(cfg)
    model.load_state_dict(ckpt['model'])
    encoder, model = encoder.to(args.device).eval(), model.to(args.device).eval()

    # probes are fitted on pairs from training clips (1.5x the largest size, as some pairs lose the ball);
    # evaluation uses held-out clips (val split, or all of --data), one pair per clip at a fixed start frame
    clips = list_clips(data_cfg['path'])
    train_clips = clips[:-data_cfg['val_clips']]
    train_idx = torch.randperm(len(train_clips), generator=gen)[:int(1.5 * max(args.sizes))].tolist()
    path = args.data or data_cfg['path']
    val_clips = list_clips(path) if args.data else clips[-data_cfg['val_clips']:]
    idx = torch.linspace(0, len(val_clips) - 1, min(args.num, len(val_clips))).long().unique().tolist()
    make = lambda p, c: VideoPairDataset(p, c, data_cfg['frame_gap'], data_cfg['image_size'], random_start=False)

    print(f"encoding {len(train_idx)} training pairs")
    tr_mu, tr_pos, tr_init = load(model, encoder, make(data_cfg['path'], train_clips), train_idx, args.device,
                                  args.batch_size, args.num_workers)
    print(f"encoding {len(idx)} held-out pairs")
    mu, pos, init = load(model, encoder, make(path, val_clips), idx, args.device, args.batch_size, args.num_workers)

    tr_y, tr_vis = targets(tr_pos, tr_init)
    y, vis = targets(pos, init)
    tr_mu, tr_y = tr_mu[tr_vis], {k: v[tr_vis] for k, v in tr_y.items()}
    mu, y = mu[vis], {k: v[vis] for k, v in y.items()}
    sizes = [n for n in args.sizes if n <= len(tr_mu)]
    print(f"visible pairs: {len(tr_mu)} / {len(tr_vis)} train, {len(mu)} / {len(vis)} held-out | "
          f"latent: {mu.shape[1]} dims, per-dim std {tr_mu.std(0).min():.3g} .. {tr_mu.std(0).max():.3g}")
    if len(sizes) < len(args.sizes):
        print(f"skipping sizes {[n for n in args.sizes if n > len(tr_mu)]}: only {len(tr_mu)} visible training pairs")

    chance = errors({k: v.mean(0, keepdim=True).expand_as(y[k]) for k, v in tr_y.items()}, y)
    runs = []  # per size: list over repeats of {curve: error}
    ratios = []
    for n in sizes:
        runs.append([])
        for _ in range(args.repeats if n < len(tr_mu) else 1):
            sub = torch.randperm(len(tr_mu), generator=gen)[:n]
            probes = {k: fit_ridge(tr_mu[sub], v[sub]) for k, v in tr_y.items()}
            runs[-1].append(errors({k: ridge_predict(p, mu) for k, p in probes.items()}, y))
            ratios.append({'n': n, **{k: p['ratio'] for k, p in probes.items()}})
    curves = list(chance)
    stats = {c: {'mean': [sum(r[c] for r in rs) / len(rs) for rs in runs],
                 'std': [torch.tensor([r[c] for r in rs]).std(unbiased=False).item() for rs in runs]} for c in curves}

    units = {'position': 'px^2', 'position_prev': 'px^2', 'velocity_x': 'px^2', 'velocity_y': 'px^2', 'diameter': 'px^2'}
    print(f"\nmean squared error on held-out pairs (rmse in brackets)\n{'n':>6} | " + ' | '.join(f"{c:>18}" for c in curves))
    for i, n in enumerate(sizes):
        print(f"{n:>6} | " + ' | '.join(f"{stats[c]['mean'][i]:>9.3g} ({stats[c]['mean'][i] ** 0.5:>5.2f})" for c in curves))
    print(f"{'chance':>6} | " + ' | '.join(f"{chance[c]:>9.3g} ({chance[c] ** 0.5:>5.2f})" for c in curves))

    out = args.out or os.path.join(os.path.dirname(args.checkpoint), 'ablations',
                                   'latent_probe_' + os.path.splitext(os.path.basename(args.checkpoint))[0]
                                   + (f"_{os.path.splitext(os.path.basename(args.data))[0]}" if args.data else ''))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'results.json'), 'w') as f:
        json.dump({'checkpoint': args.checkpoint, 'data': path, 'units': units, 'sizes': sizes, 'chance': chance,
                   'num_train_visible': len(tr_mu), 'num_eval_visible': len(mu), 'ridge_ratios': ratios,
                   **{c: {**stats[c], 'per_repeat': [[r[c] for r in rs] for rs in runs]} for c in curves}}, f, indent=1)

    panels = [('a. position', 'squared error $(x-\\hat{x})^2 + (y-\\hat{y})^2$  [px$^2$]',
               [('position', '$p_t$ (target frame)', '#2a78d6'), ('position_prev', '$p_{t-1}$ (context frame)', '#eb6834')]),
              ('b. velocity $p_t - p_{t-1}$', 'squared error per direction  [px$^2$]',
               [('velocity_x', '$v_x$', '#2a78d6'), ('velocity_y', '$v_y$', '#eb6834')]),
              ('c. diameter', 'squared error $(d-\\hat{d})^2$  [px$^2$]', [('diameter', 'diameter', '#2a78d6')])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (title, ylabel, lines) in zip(axes, panels):
        for c, label, color in lines:
            m, s = torch.tensor(stats[c]['mean']), torch.tensor(stats[c]['std'])
            ax.plot(sizes, m, '-o', color=color, linewidth=2, markersize=4, label=f'probe on latent, {label}')
            ax.fill_between(sizes, (m - s).clamp_min(m.min() * 1e-2), m + s, color=color, alpha=0.2, linewidth=0)
            ax.axhline(chance[c], color=color, linestyle='--', linewidth=1.2, alpha=0.7, label=f'chance (mean), {label}')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('training pairs for the probe')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, which='both')
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f'Linear probes on the latent action ({mu.shape[1]}-d code), {len(mu)} held-out pairs', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'probe_scaling.png'), dpi=120)
    plt.close(fig)
    print(f"saved to {out}")


if __name__ == '__main__':
    main()
