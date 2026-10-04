""" Ablation: how errors compound over a rollout with the latent action vs the real action (parabola data).

Actions per step k (codes from velocity_action.py's probe.pt, fitted on training pairs):
    latent     mu = IDM(s_{k-1}, s_k) on ground-truth frames
    dpos       W dpos_k + b, the true displacement p_k - p_{k-1} in px            (real action)
    dpos_pos   W [dpos_k, p_{k-1}] + b                                           (real action + position)
    mean       the mean code (no information)
Each action is run two ways from the same noise:
    autoregressive   s^_k = WM(s^_{k-1}, a_k), from the true s_0 (errors compound)
    teacher-forced   s^_k = WM(s_{k-1}, a_k), from the true previous frame (one-step error only)
compounding at step k = autoregressive error - teacher-forced error. copy: s_0 (autoregressive) / s_{k-1} (teacher-forced).
Errors per step, on held-out clips, only at steps with the ball on screen:
    ball position error   L2 in px between the true ball centre and the one a linear probe (ridge on the standardised,
                          flattened features, fitted on real s_t of --num_pos_train training pairs) reads off s^_k;
                          'probe floor' is the same probe on the true s_k
    ball cosine           cos(s^_k, s_k) over the patches the ball covers at k-1 or k

    python ablations/error_compounding.py runs/parabola/step_10000.pt --data dataset/parabola_eval.hdf5 \
        --probe runs/parabola/ablations/velocity_action_step_10000_parabola_eval_h7/probe.pt [--horizon 7] [--num 500]

Writes to <checkpoint dir>/ablations/error_compounding_<checkpoint name>[_<data name>]_h<horizon>/:
    results.json, compounding.png
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ablations.velocity_action import FRAME, action_inputs, ball_mask, batches
from src.data import VideoPairDataset, list_clips
from src.utils import fit_ridge, ridge_predict
from train import build_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--probe', required=True, help="velocity_action.py's probe.pt (action codes from dpos)")
    parser.add_argument('--data', default=None, help='hdf5 to evaluate on, all clips (default: val split of the training file)')
    parser.add_argument('--horizon', type=int, default=7, help='rollout steps (<= 31 // frame_gap)')
    parser.add_argument('--num', type=int, default=500, help='held-out clips')
    parser.add_argument('--num_pos_train', type=int, default=2000, help='training pairs for the ball-position probe')
    parser.add_argument('--steps', type=int, default=None, help='Euler steps (default: the config)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    H, cuda = args.horizon, args.device.startswith('cuda')

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg, data_cfg = ckpt['config'], ckpt['config']['data']
    encoder, model = build_models(cfg)
    model.load_state_dict(ckpt['model'])
    encoder, model = encoder.to(args.device).eval(), model.to(args.device).eval()
    steps = args.steps or cfg['flow']['sample_steps']
    grid = data_cfg['image_size'] // encoder.patch_size
    make = lambda p, c, T: VideoPairDataset(p, c, data_cfg['frame_gap'], data_cfg['image_size'], random_start=False, num_frames=T)
    flat = lambda f: model.normalize(f).flatten(-2).half()  # ..., P, C raw -> ..., P * C standardised, the probe's input

    probe = torch.load(args.probe, map_location='cpu')
    maps = {k: (m['W'].to(args.device), m['b'].to(args.device)) for k, m in probe['maps'].items()}
    mean_code = probe['mean_code'].to(args.device)
    up = model.idm.bottleneck.up

    # ---- ball-position probe on real features of training pairs (s_t -> ball px) ----
    clips = list_clips(data_cfg['path'])
    num_train = len(clips) - data_cfg['val_clips']
    train_idx = torch.randperm(num_train, generator=torch.Generator().manual_seed(args.seed + 1))[:args.num_pos_train]
    xs, ys = [], []
    for _, feats, _, px, _ in batches(model, encoder, make(data_cfg['path'], clips[:-data_cfg['val_clips']], 2),
                                      train_idx.tolist(), args, visible_only=True):
        xs.append(flat(feats[:, 1]))
        ys.append(px[:, 1])
    pos_probe = fit_ridge(torch.cat(xs), torch.cat(ys))
    del xs
    torch.cuda.empty_cache()
    print(f"ball-position probe on {len(torch.cat(ys))} pairs: LOO rmse {pos_probe['loo_mse'] ** 0.5:.2f} px per coordinate")

    # ---- autoregressive and teacher-forced rollouts on held-out clips ----
    path = args.data or data_cfg['path']
    val_clips = list_clips(path) if args.data else clips[-data_cfg['val_clips']:]
    idx = torch.linspace(0, len(val_clips) - 1, min(args.num, len(val_clips))).long().unique().tolist()
    names = ['latent', *maps, 'mean', 'copy']
    modes = ['ar', 'tf']
    pos_sum = {(k, m): torch.zeros(H) for k in names for m in modes}
    cos_sum = {(k, m): torch.zeros(H) for k in names for m in modes}
    floor_sum, count, n = torch.zeros(H), torch.zeros(H), 0
    print(f"rolling out {len(idx)} held-out clips, {H} steps")
    for i, (_, feats, mu, px, radius) in enumerate(batches(model, encoder, make(path, val_clips, H + 1), idx, args,
                                                            visible_only=False)):
        bs, target, true_px = len(feats), feats[:, 1:], px[:, 1:]  # bs, H, ...
        on_screen = ((true_px >= 0) & (true_px < FRAME)).all(-1).float()  # bs, H
        mask = torch.stack([ball_mask(px[:, h:h + 2], radius, grid) for h in range(H)], dim=1).float()  # bs, H, P
        pos_err = lambda pred: (ridge_predict(pos_probe, flat(pred).flatten(0, 1)).unflatten(0, (bs, H)) - true_px).norm(dim=-1)
        inputs = action_inputs(px)
        codes = {'latent': mu, **{k: inputs[k] @ W + b for k, (W, b) in maps.items()}, 'mean': mean_code.expand_as(mu)}

        for k in names:
            if k == 'copy':
                preds = {'ar': feats[:, :1].expand(-1, H, -1, -1), 'tf': feats[:, :-1]}
            else:
                action = lambda h: up(codes[k][:, h]).unsqueeze(1)  # bs, 1, latent_dim
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=cuda):
                    torch.manual_seed(args.seed + i)
                    context, ar = feats[:, 0], []
                    for h in range(H):
                        context = model.predict(context, action(h), steps).float()
                        ar.append(context)
                    torch.manual_seed(args.seed + i)
                    tf = [model.predict(feats[:, h], action(h), steps).float() for h in range(H)]
                preds = {'ar': torch.stack(ar, dim=1), 'tf': torch.stack(tf, dim=1)}  # bs, H, P, C
            for m, pred in preds.items():
                c = F.cosine_similarity(pred, target, dim=-1)  # bs, H, P
                cos_ball = (c * mask).sum(-1) / mask.sum(-1).clamp_min(1)
                pos_sum[k, m] += (pos_err(pred) * on_screen).sum(0).cpu()
                cos_sum[k, m] += (cos_ball * on_screen).sum(0).cpu()
            del preds
        floor_sum += (pos_err(target) * on_screen).sum(0).cpu()
        count += on_screen.sum(0).cpu()
        n += bs
        if (i + 1) % 5 == 0 and cuda:
            print(f"  {n} clips, peak GPU {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB", flush=True)

    # steps where no clip has the ball on screen (late steps: it has fallen out) have no value: None in results.json
    avg = lambda x: [v if c > 0 else None for v, c in zip((x / count.clamp_min(1)).tolist(), count.tolist())]
    pos = {f'{k}/{m}': avg(pos_sum[k, m]) for k in names for m in modes}
    cos = {f'{k}/{m}': avg(cos_sum[k, m]) for k in names for m in modes}
    floor = avg(floor_sum)
    H = int((count > 0).sum())  # report and plot the steps that have clips
    pos, cos, floor = {k: v[:H] for k, v in pos.items()}, {k: v[:H] for k, v in cos.items()}, floor[:H]

    print(f"\n{n} held-out clips, {steps} Euler steps; clips with the ball on screen per step: {count.long().tolist()}")
    print(f"ball position error [px], autoregressive / teacher-forced (probe floor on true s_k in the last column)")
    print(f"{'step':>4} | " + ' | '.join(f"{k:>13}" for k in names) + f" | {'floor':>5}")
    for h in range(H):
        print(f"{h + 1:>4} | " + ' | '.join(f"{pos[f'{k}/ar'][h]:>5.1f} / {pos[f'{k}/tf'][h]:>5.1f}" for k in names)
              + f" | {floor[h]:>5.1f}")
    print(f"\nball-patch cosine, autoregressive / teacher-forced")
    for h in range(H):
        print(f"{h + 1:>4} | " + ' | '.join(f"{cos[f'{k}/ar'][h]:.3f} / {cos[f'{k}/tf'][h]:.3f}" for k in names))

    out = os.path.join(os.path.dirname(args.checkpoint), 'ablations',
                       'error_compounding_' + os.path.splitext(os.path.basename(args.checkpoint))[0]
                       + (f"_{os.path.splitext(os.path.basename(args.data))[0]}" if args.data else '') + f'_h{args.horizon}')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'results.json'), 'w') as f:
        json.dump({'checkpoint': args.checkpoint, 'data': path, 'probe': args.probe, 'num_eval': n, 'horizon': args.horizon,
                   'on_screen_count': count.tolist(), 'pos_probe_loo_rmse': pos_probe['loo_mse'] ** 0.5,
                   'ball_pos_error_px': pos, 'ball_cos': cos, 'probe_floor_px': floor}, f, indent=1)

    colors = {'latent': '#2a78d6', 'dpos': '#eb6834', 'dpos_pos': '#3aa35b', 'mean': '#8a8a86', 'copy': '#c4c4c0'}
    labels = {'latent': 'latent (IDM)', 'dpos': r'real $\Delta p$', 'dpos_pos': r'real $[\Delta p, p_{k-1}]$',
              'mean': 'mean (no info)', 'copy': 'copy'}
    ks = range(1, H + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    for k in names:
        style = dict(color=colors[k], linewidth=2, markersize=4)
        axes[0].plot(ks, pos[f'{k}/ar'], '-o', label=f'{labels[k]}', **style)
        axes[0].plot(ks, pos[f'{k}/tf'], '--', alpha=0.7, **style)
        if k != 'copy':  # its gap is s_0 vs s_{k-1}, not the model's compounding (and would squash the axis)
            axes[1].plot(ks, [a - t for a, t in zip(pos[f'{k}/ar'], pos[f'{k}/tf'])], '-o', label=labels[k], **style)
        axes[2].plot(ks, cos[f'{k}/ar'], '-o', label=labels[k], **style)
        axes[2].plot(ks, cos[f'{k}/tf'], '--', alpha=0.7, **style)
    axes[0].plot(ks, floor, ':', color='k', linewidth=1.5, label='probe floor (true $s_k$)')
    axes[1].axhline(0, color='k', linewidth=0.8)
    titles = ['ball position error  (solid: autoregressive, dashed: teacher-forced)',
              'compounding: autoregressive - teacher-forced', 'ball-patch cosine  (solid: AR, dashed: TF)']
    ylabels = ['L2 error [px]', 'extra L2 error [px]', r'cos$(\hat{s}_k, s_k)$ on ball patches']
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_xlabel('rollout step k')
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(ks))
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f'Error compounding, latent vs real action ({n} held-out clips)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'compounding.png'), dpi=120)
    plt.close(fig)
    print(f"saved to {out}")


if __name__ == '__main__':
    main()
