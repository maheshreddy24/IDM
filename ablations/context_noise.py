""" Ablation: does the world model read the scene from its context s_{t-1}, or from the latent action a_t?

The IDM sees the clean pair, a_t = IDM(s_{t-1}, s_t); one input of the world model is corrupted:
    --target context   pred = WM(s_{t-1} + sigma * std_s * eps, a_t)     std_s: per-channel feature std from training
    --target action    pred = WM(s_{t-1}, a_t + sigma * std_a * eps)     std_a: per-dim std of a_t over the probe clips
eps ~ N(0, I); 'noise' replaces the input with pure noise (mean + std * eps). The metric is where the balls are in pred:
a linear probe (ridge on the flattened, standardised patch tokens -> [left x, left y, right x, right y], world units)
reads the positions off pred, and we report the L2 error per ball against the ground truth at t:
    fixed probe      fitted once on real s_t of training clips, applied to pred at every sigma
    per-sigma probe  refitted at each sigma on the world model's predictions for the training clips (same corruption),
                     so it still works if pred drifts away from real features: is the position recoverable at all?
    noisy input      (context only) fixed probe on the corrupted s_{t-1} vs the positions at t-1: how much the
                     corrupted context itself still shows
Baselines: fixed probe on the true s_t (probe floor), on s_{t-1} (copy: the model returns its context) and the
training mean position (chance). If the error stays low under a pure-noise context, a_t carries the ball positions
(leakage); if it climbs to chance, the world model reads the scene from the context. Action noise is the converse.
Also printed: cos(s_t, pred), the per-token cosine used before.

    python ablations/context_noise.py runs/toy/step_10000.pt [--target context action] [--num_train 2000] [--num 256]
        [--data EVAL.hdf5] [--no_per_sigma] [--device cuda]

Writes to <checkpoint dir>/ablations/context_noise_<checkpoint name>[_<data name>]/:
    results.json, position_error.png, samples_<target>.png
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


@torch.no_grad()
def load(dataset, idx, encoder, device, bs, num_workers):
    """ -> feats N, 2, P, C raw features of (s_{t-1}, s_t) (float16), positions N, 2, 2, 2 (frame, ball, xy) """
    feats, positions = [], []
    for batch in DataLoader(Subset(dataset, idx), batch_size=bs, num_workers=num_workers):
        if 'positions' not in batch:
            raise ValueError(f'{dataset.path} has no position_streams')
        feats.append(encode_pairs(encoder, batch['frames'].to(device)).half())
        positions.append(batch['positions'])
    return torch.cat(feats), torch.cat(positions).to(device)


@torch.no_grad()
def infer_actions(model, feats, bs):
    """ feats N, 2, P, C -> a_t N, 1, latent_dim from the clean pairs """
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return torch.cat([model.infer_action(feats[i:i + bs].float()).float() for i in range(0, len(feats), bs)])


def probe_input(model, feats, bs=256):
    """ raw features N, P, C -> standardised and flattened N, P * C (float16), the probe's input """
    return torch.cat([model.normalize(feats[i:i + bs].float()).flatten(1).half() for i in range(0, len(feats), bs)])


def corrupt(x, level, mean, std, seed):
    """ x + level * std * eps, or mean + std * eps for level 'noise'; eps is fixed by seed, so shared across levels """
    eps = torch.randn(x.shape, generator=torch.Generator(x.device).manual_seed(seed), device=x.device)
    return mean + std * eps if level == 'noise' else x + level * std * eps


@torch.no_grad()
def predict(model, feats, actions, target, level, stats, steps, bs, seed):
    """ WM prediction of s_t for every pair, with `target` ('context' or 'action') corrupted at `level`
    -> preds N, P, C and the (possibly corrupted) contexts N, P, C, both float16 """
    preds, contexts = [], []
    for i in range(0, len(feats), bs):
        context, action = feats[i:i + bs, 0].float(), actions[i:i + bs]
        if target == 'context':
            context = corrupt(context, level, *stats['context'], seed + i)
        else:
            action = corrupt(action, level, *stats['action'], seed + i)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            preds.append(model.predict(context, action, steps).half())
        contexts.append(context.half())
    return torch.cat(preds), torch.cat(contexts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--target', nargs='+', default=['context', 'action'], choices=['context', 'action'],
                        help='world-model input to corrupt; one sweep per target')
    parser.add_argument('--sigmas', type=float, nargs='+', default=[0, 0.1, 0.25, 0.5, 1, 2, 4],
                        help='noise levels in units of the input std; pure noise is always added')
    parser.add_argument('--num_train', type=int, default=2000, help='training clips the probes are fitted on')
    parser.add_argument('--num', type=int, default=256, help='held-out clips')
    parser.add_argument('--no_per_sigma', dest='per_sigma', action='store_false',
                        help='skip the per-sigma probes (each needs WM predictions for all training clips)')
    parser.add_argument('--steps', type=int, default=None, help='Euler steps (default: the config)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_plots', type=int, default=4, help='clips drawn in samples_<target>.png')
    parser.add_argument('--data', default=None, help='hdf5 to evaluate on, all clips (default: val split of the training file)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg, data_cfg = ckpt['config'], ckpt['config']['data']
    steps = args.steps or cfg['flow']['sample_steps']
    encoder, model = build_models(cfg)
    model.load_state_dict(ckpt['model'])
    encoder, model = encoder.to(args.device).eval(), model.to(args.device).eval()

    # probes are fitted on pairs from training clips; evaluation uses held-out clips (val split, or all of --data)
    clips = list_clips(data_cfg['path'])
    train_set = VideoPairDataset(data_cfg['path'], clips[:-data_cfg['val_clips']], data_cfg['frame_gap'],
                                 data_cfg['image_size'], random_start=False)
    path = args.data or data_cfg['path']
    val_clips = list_clips(path) if args.data else clips[-data_cfg['val_clips']:]
    val_set = VideoPairDataset(path, val_clips, data_cfg['frame_gap'], data_cfg['image_size'], random_start=False)
    train_idx = torch.linspace(0, len(train_set) - 1, args.num_train).long().tolist()
    idx = torch.linspace(0, len(val_set) - 1, args.num).long().tolist()

    load_args = (encoder, args.device, args.batch_size, data_cfg['num_workers'])
    tr_feats, tr_pos = load(train_set, train_idx, *load_args)
    feats, pos = load(val_set, idx, *load_args)
    tr_actions, actions = infer_actions(model, tr_feats, args.batch_size), infer_actions(model, feats, args.batch_size)
    stats = {'context': (model.feat_mean, model.feat_std), 'action': (tr_actions.mean(0), tr_actions.std(0))}
    print(f"{len(train_idx)} probe clips, {len(idx)} eval clips; action std {tr_actions.std(0).mean():.3f}")

    # probe targets [left x, left y, right x, right y] at t; error = L2 per ball -> N, 2 (left, right)
    ball_error = lambda out, p: (out.view(-1, 2, 2) - p).norm(dim=-1)
    probe_error = lambda probe, f, p: ball_error(ridge_predict(probe, probe_input(model, f)), p)
    fixed = fit_ridge(probe_input(model, tr_feats[:, 1]), tr_pos[:, 1].flatten(1))
    baselines = {
        'floor': probe_error(fixed, feats[:, 1], pos[:, 1]),  # true s_t
        'copy': probe_error(fixed, feats[:, 0], pos[:, 1]),  # s_{t-1}
        'chance': ball_error(tr_pos[:, 1].flatten(1).mean(0, keepdim=True), pos[:, 1]),
    }
    print(f"fixed probe: ridge ratio {fixed['ratio']:g}, LOO rmse {fixed['loo_mse'] ** 0.5:.3f} | eval error (mean L2 per ball): "
          + ' | '.join(f"{k} {v.mean():.3f}" for k, v in baselines.items()))

    levels = args.sigmas + ['noise']
    labels = [f'{s:g}' for s in args.sigmas] + ['noise']
    k = min(args.num_plots, len(idx))
    results, samples = {}, {}
    for target in args.target:
        res, samples[target] = {'fixed': [], 'per_sigma': [], 'noisy_input': [], 'cos': []}, []
        print(f"\n--- {target} noise ---\n{'sigma':>6} | {'fixed probe':>11} | {'per-sigma':>9} | {'noisy input':>11} | {'cos(s_t, pred)':>14}")
        for level, label in zip(levels, labels):
            pred, context = predict(model, feats, actions, target, level, stats, steps, args.batch_size, args.seed)
            res['fixed'].append(probe_error(fixed, pred, pos[:, 1]))
            res['cos'].append(F.cosine_similarity(pred.float(), feats[:, 1].float(), dim=-1).mean(-1))
            if target == 'context':
                res['noisy_input'].append(probe_error(fixed, context, pos[:, 0]))
            if args.per_sigma:
                tr_pred, _ = predict(model, tr_feats, tr_actions, target, level, stats, steps, args.batch_size, args.seed + 10 ** 6)
                probe = fit_ridge(probe_input(model, tr_pred), tr_pos[:, 1].flatten(1))
                del tr_pred
                res['per_sigma'].append(probe_error(probe, pred, pos[:, 1]))
            samples[target].append(pred[:k].float().cpu())
            row = [res[m][-1].mean().item() if res[m] else float('nan') for m in ('fixed', 'per_sigma', 'noisy_input', 'cos')]
            print(f"{label:>6} | {row[0]:>11.3f} | {row[1]:>9.3f} | {row[2]:>11.3f} | {row[3]:>14.4f}")
        results[target] = {m: torch.stack(v, dim=1).cpu() for m, v in res.items() if v}  # N, S(, 2)

    out = args.out or os.path.join(os.path.dirname(args.checkpoint), 'ablations',
                                   'context_noise_' + os.path.splitext(os.path.basename(args.checkpoint))[0]
                                   + (f"_{os.path.splitext(os.path.basename(args.data))[0]}" if args.data else ''))
    os.makedirs(out, exist_ok=True)
    summary = lambda v: {'mean': v.mean(-1).mean(0).tolist(),
                         'left': v[..., 0].mean(0).tolist(), 'right': v[..., 1].mean(0).tolist(), 'per_clip': v.tolist()}
    with open(os.path.join(out, 'results.json'), 'w') as f:
        json.dump({'checkpoint': args.checkpoint, 'data': path, 'sigma': labels, 'clips': [val_clips[i] for i in idx],
                   'units': 'L2 position error per ball (world units); per_clip is clips x [sigma x] (left, right)',
                   'probe': {'num_train': len(train_idx), 'ridge_ratio': fixed['ratio'], 'loo_mse': fixed['loo_mse']},
                   'baselines': {m: summary(v.cpu()) for m, v in baselines.items()},
                   **{t: {m: {'mean': v.mean(0).tolist(), 'per_clip': v.tolist()} if m == 'cos' else summary(v)
                          for m, v in r.items()} for t, r in results.items()}}, f, indent=1)

    # position error vs noise level, one panel per target (categorical x, since the last level is pure noise)
    xs = range(len(levels))
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4), squeeze=False, sharey=True)
    for ax, (target, r) in zip(axes[0], results.items()):
        for m, color, style in (('floor', '#8a8a86', ':'), ('copy', '#8a8a86', '--'), ('chance', '#c4c4c0', '-')):
            ax.axhline(baselines[m].mean().item(), color=color, linestyle=style, linewidth=2,
                       label={'floor': r'probe on true $s_t$', 'copy': r'probe on $s_{t-1}$ (copy)', 'chance': 'mean position'}[m])
        lines = [('fixed', '#2a78d6', '-', r'fixed probe on $\hat{s}_t$'), ('per_sigma', '#eb6834', '-', r'per-$\sigma$ probe on $\hat{s}_t$'),
                 ('noisy_input', '#2a78d6', '-.', r'fixed probe on noisy $s_{t-1}$ (vs pos at $t-1$)')]
        for m, color, style, label in lines:
            if m in r:
                ax.plot(xs, r[m].mean(-1).mean(0), style, color=color, linewidth=2, marker='o', markersize=5, label=label)
        ax.set_xticks(list(xs), labels)
        ax.set_xlabel(rf'{target} noise $\sigma$ (× std; "noise" = {target} replaced)')
        ax.set_title(f'{target} noise ({len(idx)} held-out clips)', fontsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc='upper left')
    axes[0, 0].set_ylabel('ball position error (L2, world units)')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'position_error.png'), dpi=120)
    plt.close(fig)

    # per target, rows: clips; columns: frames, PCA of the target, then the prediction at each noise level
    grid, size = (data_cfg['image_size'] // encoder.patch_size,) * 2, data_cfg['image_size']
    images = torch.stack([val_set[i]['frames'] for i in idx[:k]])  # k, 2, 3, h, w
    frames = denormalize(images.flatten(0, 1)).unflatten(0, (k, 2))
    context, target_feats = feats[:k, 0].float().cpu(), feats[:k, 1].float().cpu()
    for target, preds in samples.items():
        preds = torch.stack(preds, dim=1)  # k, S, P, C
        pca = fit_pca(torch.cat([context, target_feats, preds.flatten(0, 1)]).flatten(0, 1))
        panels = torch.cat([frames, pca_rgb(target_feats, pca, grid, size)[:, None],
                            pca_rgb(preds.flatten(0, 1), pca, grid, size).unflatten(0, preds.shape[:2])], dim=1)
        plot_grid(panels, os.path.join(out, f'samples_{target}.png'),
                  col_titles=['$s_{t-1}$', '$s_t$', 'PCA $s_t$'] + [f'{target} σ={l}' for l in labels])
    print(f"saved to {out}")


if __name__ == '__main__':
    main()
