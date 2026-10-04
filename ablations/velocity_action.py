""" Ablation: replace the latent action with the ball's true displacement (parabola data), in rollouts.

Linear maps to the latent code (ridge) are fitted on pairs from a random --train_frac of the training clips (only pairs
with the ball on screen), from dpos = p_t - p_{t-1} the ball's displacement in px (2 -> action_dim) and from
[dpos, p_{t-1}] (4 -> action_dim), and saved to probe.pt (reuse it with --probe). On held-out clips the world model then rolls out --horizon steps autoregressively from the
true first frame, with the action at each step given by:
    latent   the IDM's code mu = IDM(s_{k-1}, s_k) on consecutive ground-truth frames (the normal setting)
    dpos     the code mapped from the true displacement, W dpos + b
    dpos_pos the code mapped from the true displacement and the true position at k-1, W [dpos, p_{k-1}] + b
    mean     the mean code (no information: what the world model does without an action)
Per step we report cos(pred, s_k) over all tokens and over the ball's patches (patches the ball covers at k-1 or k;
the background dominates the full cosine; steps with the ball off screen are left out of it). copy: pred = s_0.

    python ablations/velocity_action.py runs/parabola/step_10000.pt --data dataset/parabola_eval.hdf5 [--train_frac 0.4]
        [--probe PROBE.pt] [--fit_only] [--horizon 7] [--num 1000]

Writes to <checkpoint dir>/ablations/velocity_action_<checkpoint name>[_<data name>]_h<horizon>/:
    probe.pt, results.json, rollout_cos.png, sample_<i>.png (frames and PCA of GT / each action's rollout)
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import VideoPairDataset, list_clips
from src.utils import denormalize, fit_pca, fit_ridge, pca_rgb, plot_grid, ridge_predict
from train import build_models, encode_pairs

PX, FRAME = 25.6, 256  # pixels per world unit, video frame size (the 10 x 10 world fills it, y up)


def batches(model, encoder, dataset, idx, args, visible_only):
    """ yields frames bs, T, 3, h, w (normalised, CPU), feats bs, T, P, C, mu bs, T-1, action_dim (IDM on consecutive
    frames), ball px bs, T, 2 (xy, y down), radius px bs; visible_only keeps clips with the ball on screen in every frame """
    loader = DataLoader(Subset(dataset, idx), batch_size=args.batch_size, num_workers=args.num_workers)
    for i, batch in enumerate(loader):
        pos = batch['positions']  # bs, T, 2 world units
        px = torch.stack([pos[..., 0] * PX, FRAME - pos[..., 1] * PX], dim=-1)
        keep = ((px >= 0) & (px < FRAME)).flatten(1).all(1) if visible_only else torch.ones(len(px), dtype=torch.bool)
        if (i + 1) % 100 == 0:
            mem = f", peak GPU {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB" if args.device.startswith('cuda') else ''
            print(f"  {(i + 1) * args.batch_size} / {len(idx)} clips{mem}", flush=True)
        if not keep.any():
            continue
        # encoder (keeps every layer's hidden states) and IDM (full attention matrix) in chunks of ~batch_size images / pairs
        frames, chunk = batch['frames'][keep], max(1, args.batch_size // px.shape[1])
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.device.startswith('cuda')):
            feats = torch.cat([encode_pairs(encoder, frames[j:j + chunk].to(args.device)) for j in range(0, len(frames), chunk)])
            pairs = torch.stack([feats[:, :-1], feats[:, 1:]], dim=2).flatten(0, 1)  # bs (T-1), 2, P, C
            mu = torch.cat([model.idm(model.normalize(pairs[j:j + args.batch_size]))['mu'].float()
                            for j in range(0, len(pairs), args.batch_size)]).unflatten(0, (len(feats), -1)).squeeze(2)
        yield batch['frames'][keep], feats, mu, px[keep].to(args.device), (PX * batch['init'][keep, 0]).to(args.device)


def action_inputs(px):
    """ ball px bs, T, 2 -> {map name: bs, T-1, D}, the inputs for the steps k = 1 .. T-1 """
    dpos = px[:, 1:] - px[:, :-1]
    return {'dpos': dpos, 'dpos_pos': torch.cat([dpos, px[:, :-1]], dim=-1)}


def ball_mask(px, radius, grid):
    """ px bs, 2, 2 (ball at k-1, k) -> bs, P bool: patches overlapping the ball at k-1 or k """
    size = FRAME / grid
    c = (torch.arange(grid, device=px.device) + 0.5) * size
    cy, cx = torch.meshgrid(c, c, indexing='ij')
    centres = torch.stack([cx, cy], dim=-1).flatten(0, 1)  # P, 2 (xy)
    dist = (centres[None, None] - px[:, :, None]).norm(dim=-1)  # bs, 2, P
    return (dist < radius[:, None, None] + size / 2).any(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--data', default=None, help='hdf5 to evaluate on, all clips (default: val split of the training file)')
    parser.add_argument('--train_frac', type=float, default=0.4, help='fraction of the training clips for the dpos -> code map')
    parser.add_argument('--probe', default=None, help='a saved probe.pt: skip fitting')
    parser.add_argument('--fit_only', action='store_true', help='stop after fitting and saving probe.pt')
    parser.add_argument('--horizon', type=int, default=7, help='rollout steps (<= 31 // frame_gap)')
    parser.add_argument('--num', type=int, default=1000, help='held-out clips')
    parser.add_argument('--num_plots', type=int, default=6, help='clips drawn as sample_<i>.png')
    parser.add_argument('--steps', type=int, default=None, help='Euler steps (default: the config)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    H = args.horizon

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg, data_cfg = ckpt['config'], ckpt['config']['data']
    encoder, model = build_models(cfg)
    model.load_state_dict(ckpt['model'])
    encoder, model = encoder.to(args.device).eval(), model.to(args.device).eval()
    steps = args.steps or cfg['flow']['sample_steps']
    grid = data_cfg['image_size'] // encoder.patch_size
    make = lambda p, c, T: VideoPairDataset(p, c, data_cfg['frame_gap'], data_cfg['image_size'], random_start=False, num_frames=T)
    path = args.data or data_cfg['path']
    out = os.path.join(os.path.dirname(args.checkpoint), 'ablations',
                       'velocity_action_' + os.path.splitext(os.path.basename(args.checkpoint))[0]
                       + (f"_{os.path.splitext(os.path.basename(args.data))[0]}" if args.data else '') + f'_h{H}')
    os.makedirs(out, exist_ok=True)

    # ---- dpos -> code on training pairs (or a saved one) ----
    clips = list_clips(data_cfg['path'])
    if args.probe:
        probe = torch.load(args.probe, map_location='cpu')
        print(f"loaded {args.probe}: fitted on {probe['num_train']} pairs, R^2 "
              + ', '.join(f"{k} {m['r2']:.3f}" for k, m in probe['maps'].items()))
    else:
        num_train = len(clips) - data_cfg['val_clips']
        train_idx = torch.randperm(num_train, generator=torch.Generator().manual_seed(args.seed))[:int(args.train_frac * num_train)]
        print(f"encoding {len(train_idx)} training pairs ({args.train_frac:.0%} of {num_train} clips)")
        mus, pxs = [], []
        for _, _, mu, px, _ in batches(model, encoder, make(data_cfg['path'], clips[:-data_cfg['val_clips']], 2),
                                       train_idx.tolist(), args, visible_only=True):
            mus.append(mu[:, 0])
            pxs.append(px)
        mus, pxs = torch.cat(mus), torch.cat(pxs)
        mean_code = mus.mean(0)
        probe = {'maps': {}, 'mean_code': mean_code.cpu(), 'num_train': len(mus), 'checkpoint': args.checkpoint}
        for k, x in action_inputs(pxs).items():
            fit = fit_ridge(x[:, 0], mus)
            r2 = 1 - (ridge_predict(fit, x[:, 0]) - mus).pow(2).sum() / (mus - mean_code).pow(2).sum()
            probe['maps'][k] = {'W': fit['W'].cpu(), 'b': fit['b'].cpu(), 'r2': r2.item(), 'ratio': fit['ratio']}
            print(f"{k} -> code fitted on {len(mus)} pairs; explains {r2:.3f} of the code's variance (train)")
        torch.save(probe, os.path.join(out, 'probe.pt'))
        print(f"saved {os.path.join(out, 'probe.pt')}")
        if args.fit_only:
            return
        del mus, pxs
        torch.cuda.empty_cache()
    maps = {k: (m['W'].to(args.device), m['b'].to(args.device)) for k, m in probe['maps'].items()}
    mean_code = probe['mean_code'].to(args.device)

    # ---- rollouts on held-out clips with each action ----
    val_clips = list_clips(path) if args.data else clips[-data_cfg['val_clips']:]
    idx = torch.linspace(0, len(val_clips) - 1, min(args.num, len(val_clips))).long().unique().tolist()
    names = ['latent', *maps, 'mean', 'copy']
    cos_sum, ball_sum, ball_count, n = {k: torch.zeros(H) for k in names}, {k: torch.zeros(H) for k in names}, torch.zeros(H), 0
    up, samples = model.idm.bottleneck.up, None
    print(f"rolling out {len(idx)} held-out clips, {H} steps")
    for i, (frames, feats, mu, px, radius) in enumerate(batches(model, encoder, make(path, val_clips, H + 1), idx, args,
                                                                 visible_only=False)):
        inputs = action_inputs(px)
        codes = {'latent': mu, **{k: inputs[k] @ W + b for k, (W, b) in maps.items()},
                 'mean': mean_code.expand_as(mu)}  # bs, H, action_dim
        target = feats[:, 1:]
        mask = torch.stack([ball_mask(px[:, h:h + 2], radius, grid) for h in range(H)], dim=1).float()  # bs, H, P
        mask = mask * ((px[:, 1:] >= 0) & (px[:, 1:] < FRAME)).all(-1)[..., None]  # ball on screen at k
        first = samples is None
        if first:
            j = slice(0, args.num_plots)
            samples = {'frames': frames[j], 'gt': feats[j].cpu()}
        for k in names:
            if k == 'copy':
                pred = feats[:, :1].expand(-1, H, -1, -1)
            else:
                torch.manual_seed(args.seed + i)  # same noise for every action
                context, rollout = feats[:, 0], []
                for h in range(H):
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.device.startswith('cuda')):
                        context = model.predict(context, up(codes[k][:, h]).unsqueeze(1), steps).float()
                    rollout.append(context)
                pred = torch.stack(rollout, dim=1)  # bs, H, P, C
                if first:
                    samples[k] = pred[:args.num_plots].cpu()
            c = F.cosine_similarity(pred, target, dim=-1)  # bs, H, P
            cos_sum[k] += c.mean(-1).sum(0).cpu()
            ball_sum[k] += ((c * mask).sum(-1) / mask.sum(-1).clamp_min(1)).sum(0).cpu()
            del pred, c
        ball_count += (mask.sum(-1) > 0).sum(0).cpu()
        n += len(feats)
    cos = {k: (v / n).tolist() for k, v in cos_sum.items()}
    cos_ball = {k: (v / ball_count.clamp_min(1)).tolist() for k, v in ball_sum.items()}

    print(f"\n{n} held-out clips, {steps} Euler steps | cos(pred, s_k) all patches / ball patches")
    print(f"{'step':>4} | " + ' | '.join(f"{k:>15}" for k in names))  # all / ball
    for h in range(H):
        print(f"{h + 1:>4} | " + ' | '.join(f"{cos[k][h]:>6.4f} / {cos_ball[k][h]:.4f}" for k in names))
    with open(os.path.join(out, 'results.json'), 'w') as f:
        json.dump({'checkpoint': args.checkpoint, 'data': path, 'probe': args.probe or os.path.join(out, 'probe.pt'),
                   'probe_r2': {k: m['r2'] for k, m in probe['maps'].items()}, 'num_train': probe['num_train'], 'num_eval': n, 'horizon': H,
                   'ball_steps_counted': ball_count.tolist(), 'cos': cos, 'cos_ball': cos_ball}, f, indent=1)

    # cosine vs rollout step
    colors = {'latent': '#2a78d6', 'dpos': '#eb6834', 'dpos_pos': '#3aa35b', 'mean': '#8a8a86', 'copy': '#8a8a86'}
    labels = {'latent': 'latent action (IDM)', 'dpos': r'$\Delta p$ action', 'dpos_pos': r'$[\Delta p, p_{k-1}]$ action', 'mean': 'mean action (no info)', 'copy': r'copy $s_0$'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (vals, title) in zip(axes, [(cos, 'all patches'), (cos_ball, 'ball patches')]):
        for k in names:
            ax.plot(range(1, H + 1), vals[k], '--' if k == 'copy' else '-o', color=colors[k], linewidth=2,
                    markersize=4, alpha=0.6 if k in ('mean', 'copy') else 1, label=labels[k])
        ax.set_xlabel('rollout step k')
        ax.set_ylabel(r'mean cosine$(\hat{s}_k, s_k)$')
        ax.set_xticks(range(1, H + 1))
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f'Autoregressive rollout, {n} held-out clips', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'rollout_cos.png'), dpi=120)
    plt.close(fig)

    # per clip: frames, then PCA of the GT and of each action's rollout (one basis for all)
    size = data_cfg['image_size']
    pca = fit_pca(torch.cat([samples['gt'].flatten(0, 2)] + [samples[k].flatten(0, 2) for k in codes]))
    frames = denormalize(samples['frames'].flatten(0, 1)).unflatten(0, samples['frames'].shape[:2])  # bs, H+1, h, w, 3
    blank = torch.ones_like(frames[:, :1])
    rows = lambda x: pca_rgb(x.flatten(0, 1), pca, (grid, grid), size).unflatten(0, x.shape[:2])
    panels = [frames, rows(samples['gt'])] + [torch.cat([blank, rows(samples[k])], dim=1) for k in codes]
    for i in range(len(frames)):
        plot_grid(torch.stack([p[i] for p in panels]), os.path.join(out, f'sample_{i}.png'),
                  row_titles=['frame', 'PCA GT'] + [f'PCA {k}' for k in codes],
                  col_titles=['k=0 (given)'] + [f'k={h}' for h in range(1, H + 1)])
    print(f"saved to {out}")


if __name__ == '__main__':
    main()
