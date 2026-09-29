""" Ablation: does the latent action carry the whole next frame?

The IDM infers the action from the true last step, a_t = IDM(s_{t-1}, s_t), but the world model is given an older
context s_{t-n} (n = 1 is the normal setting): pred_n = WM(s_{t-n}, a_t). Per n we report
    cos(s_t, s_{t-n})       copy baseline: how different the true target is from the context
    cos(s_t, pred_n)        does the prediction still land on the true s_t?
    cos(s_{t-n+1}, pred_n)  does it instead advance the old context by one step?
If a_t only encodes the motion, cos(s_t, pred_n) falls with n roughly like the copy baseline while
cos(s_{t-n+1}, pred_n) stays high. If cos(s_t, pred_n) stays high as n grows, a_t carries the frame itself (leakage).

    python ablations/latent_action.py runs/toy/step_10000.pt [--max_n 7] [--num 16] [--data EVAL.hdf5] [--device cuda]

Writes to <checkpoint dir>/ablations/latent_action_<checkpoint name>[_<data name>]/: results.json, cosine.png, sample_<i>.png
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
from src.data import VideoPairDataset, list_clips
from src.utils import denormalize, fit_pca, pca_rgb, plot_grid
from train import build_models, encode_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--max_n', type=int, default=7, help='oldest context offset; <= 31 // frame_gap')
    parser.add_argument('--num', type=int, default=16, help='held-out clips')
    parser.add_argument('--num_plots', type=int, default=4, help='clips drawn as PCA figures')
    parser.add_argument('--data', default=None, help='hdf5 to evaluate on, all clips (default: val split of the training file)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    N = args.max_n

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg, data_cfg = ckpt['config'], ckpt['config']['data']
    encoder, model = build_models(cfg)
    model.load_state_dict(ckpt['model'])
    encoder, model = encoder.to(args.device).eval(), model.to(args.device).eval()

    # frames s_{t-N}, ..., s_{t-1}, s_t, frame_gap apart, from held-out clips (val split, or all of --data)
    path = args.data or data_cfg['path']
    val_clips = list_clips(path) if args.data else list_clips(path)[-data_cfg['val_clips']:]
    val_set = VideoPairDataset(path, val_clips, data_cfg['frame_gap'], data_cfg['image_size'],
                               random_start=False, num_frames=N + 1)
    idx = torch.linspace(0, len(val_set) - 1, args.num).long().tolist()
    images = torch.stack([val_set[i]['frames'] for i in idx]).to(args.device)  # bs, N+1, 3, h, w

    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.device.startswith('cuda')):
        feats = encode_pairs(encoder, images)  # bs, N+1, P, C; feats[:, N - n] = s_{t-n}
        action = model.infer_action(feats[:, -2:])  # a_t = IDM(s_{t-1}, s_t): bs, 1, latent_dim
        contexts = torch.stack([feats[:, N - n] for n in range(1, N + 1)], dim=1)  # bs, N, P, C (n = 1..N)
        preds = model.predict(contexts.flatten(0, 1), action.repeat_interleave(N, dim=0),
                              cfg['flow']['sample_steps']).float().unflatten(0, (len(idx), N))
    feats, contexts, preds = feats.cpu(), contexts.cpu(), preds.cpu()

    target = feats[:, -1:]  # s_t
    one_step = torch.stack([feats[:, N - n + 1] for n in range(1, N + 1)], dim=1)  # s_{t-n+1}
    cos = lambda a, b: F.cosine_similarity(a, b, dim=-1).mean(-1)  # -> bs, N
    results = {'copy': cos(target, contexts), 'pred_vs_target': cos(target, preds), 'pred_vs_one_step': cos(one_step, preds)}

    print(f"{'n':>3} | {'cos(s_t, s_t-n)':>15} | {'cos(s_t, pred)':>14} | {'cos(s_t-n+1, pred)':>18}")
    for n in range(1, N + 1):
        print(f"{n:>3} | {results['copy'][:, n - 1].mean():>15.4f} | {results['pred_vs_target'][:, n - 1].mean():>14.4f} | "
              f"{results['pred_vs_one_step'][:, n - 1].mean():>18.4f}")

    out = args.out or os.path.join(os.path.dirname(args.checkpoint), 'ablations',
                                   'latent_action_' + os.path.splitext(os.path.basename(args.checkpoint))[0]
                                   + (f"_{os.path.splitext(os.path.basename(args.data))[0]}" if args.data else ''))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'results.json'), 'w') as f:
        json.dump({'checkpoint': args.checkpoint, 'data': path, 'n': list(range(1, N + 1)), 'clips': [val_clips[i] for i in idx],
                   **{k: {'mean': v.mean(0).tolist(), 'per_clip': v.tolist()} for k, v in results.items()}}, f, indent=1)

    # cosine vs context offset
    ns = list(range(1, N + 1))
    fig, ax = plt.subplots(figsize=(6, 4))
    lines = [('copy', r'$\cos(s_t, s_{t-n})$  copy baseline', '#8a8a86', '--'),
             ('pred_vs_target', r'$\cos(s_t, \hat{s})$  pred vs true target', '#2a78d6', '-'),
             ('pred_vs_one_step', r'$\cos(s_{t-n+1}, \hat{s})$  pred vs one step after context', '#eb6834', '-')]
    for key, label, color, style in lines:
        ax.plot(ns, results[key].mean(0), style, color=color, linewidth=2, marker='o', markersize=5, label=label)
    ax.set_xlabel('context offset n  (world model gets $s_{t-n}$, action from $(s_{t-1}, s_t)$)')
    ax.set_ylabel('mean per-token cosine')
    ax.set_xticks(ns)
    ax.grid(alpha=0.25)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f'Latent-action leakage test ({len(idx)} held-out clips)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'cosine.png'), dpi=120)
    plt.close(fig)

    # per clip: column 0 is the target s_t, column n uses context s_{t-n}
    grid, size = (data_cfg['image_size'] // encoder.patch_size,) * 2, data_cfg['image_size']
    pca = fit_pca(torch.cat([feats.flatten(0, 2), preds.flatten(0, 2)]))
    frames = denormalize(images.flatten(0, 1).cpu()).unflatten(0, images.shape[:2])  # bs, N+1, h, w, 3
    for i in range(min(args.num_plots, len(idx))):
        ctx_frames = torch.stack([frames[i, N - n] for n in ns])
        panels = torch.stack([
            torch.cat([frames[i, -1:], ctx_frames]),
            pca_rgb(torch.cat([target[i], contexts[i]]), pca, grid, size),
            torch.cat([torch.ones_like(frames[i, :1]), pca_rgb(preds[i], pca, grid, size)]),
        ])
        plot_grid(panels, os.path.join(out, f'sample_{i}.png'), row_titles=['frame', 'PCA frame', 'PCA pred'],
                  col_titles=['target $s_t$'] + [f'ctx $s_{{t-{n}}}$\ncos(s_t, pred) {results["pred_vs_target"][i, n - 1]:.3f}'
                                                  for n in ns])
    print(f"saved to {out}")


if __name__ == '__main__':
    main()
