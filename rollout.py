""" Rollout on held-out clips. Actions come from the IDM on consecutive ground-truth frames, a_k = IDM(f_k, f_{k+1});
the world model then predicts f_{k+1} from a_k and a context that is the ground truth f_k at k = 0 and its own
previous prediction afterwards (autoregressive). --horizon 1 is the one-step rollout.

    python rollout.py runs/toy/step_10000.pt [--horizon 1] [--num 10] [--data EVAL.hdf5] [--device cuda] [--out DIR]

With frames frame_gap apart in 32-frame clips, horizon <= 31 // frame_gap (7 for frame_gap 4).
Writes to DIR (default: <checkpoint dir>/rollout_<checkpoint name>_h<horizon>[_<data name>]/):
    rollout.pt      frames bs, H+1, h, w, 3 uint8; feats_gt bs, H+1, N, C; feats_pred bs, H, N, C (for frames 1..H);
                    action bs, H, latent_dim; clips
    rollout.png     one row per sample: input frame, GT / PCA GT / PCA pred at the first step
    sample_<i>.png  (horizon > 1) per sample, over time: GT frame, PCA GT, PCA pred
PCA uses one basis fitted on all GT and predicted features, so colours are comparable across panels.
"""
import argparse
import os

import torch
import torch.nn.functional as F

from src.data import VideoPairDataset, list_clips
from src.utils import denormalize, fit_pca, pca_rgb, plot_grid
from train import build_models, encode_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--data', default=None, help='hdf5 to evaluate on, all clips (default: val split of the training file)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    H = args.horizon

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg, data_cfg = ckpt['config'], ckpt['config']['data']
    encoder, model = build_models(cfg)
    model.load_state_dict(ckpt['model'])
    encoder, model = encoder.to(args.device).eval(), model.to(args.device).eval()

    # held-out clips (val split, or all of --data) spread evenly, with a fixed start frame per clip (at horizon 1: the eval pairs)
    path = args.data or data_cfg['path']
    val_clips = list_clips(path) if args.data else list_clips(path)[-data_cfg['val_clips']:]
    val_set = VideoPairDataset(path, val_clips, data_cfg['frame_gap'], data_cfg['image_size'],
                               random_start=False, num_frames=H + 1)
    idx = torch.linspace(0, len(val_set) - 1, args.num).long().tolist()
    images = torch.stack([val_set[i]['frames'] for i in idx]).to(args.device)  # bs, H+1, 3, h, w

    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.device.startswith('cuda')):
        feats = encode_pairs(encoder, images)  # bs, H+1, N, C
        pairs = torch.stack([feats[:, :-1], feats[:, 1:]], dim=2).flatten(0, 1)  # bs H, 2, N, C
        actions = model.infer_action(pairs).float().unflatten(0, (len(idx), H))  # bs, H, 1, latent_dim
        context, preds = feats[:, 0], []
        for k in range(H):
            context = model.predict(context, actions[:, k], cfg['flow']['sample_steps']).float()
            preds.append(context)
    feats, preds, actions = feats.cpu(), torch.stack(preds, dim=1).cpu(), actions.squeeze(2).cpu()

    # per-step cosine to the ground truth; the baseline copies the first frame
    cos = F.cosine_similarity(preds, feats[:, 1:], dim=-1).mean(-1)  # bs, H
    cos_copy = F.cosine_similarity(feats[:, :1], feats[:, 1:], dim=-1).mean(-1)
    for k in range(H):
        print(f"step {k + 1}: cos(pred, gt) {cos[:, k].mean():.4f} | cos(first frame, gt) {cos_copy[:, k].mean():.4f}")

    out = args.out or os.path.join(os.path.dirname(args.checkpoint),
                                   f"rollout_{os.path.splitext(os.path.basename(args.checkpoint))[0]}_h{H}"
                                   + (f"_{os.path.splitext(os.path.basename(args.data))[0]}" if args.data else ''))
    os.makedirs(out, exist_ok=True)
    frames = denormalize(images.flatten(0, 1).cpu()).unflatten(0, images.shape[:2])  # bs, H+1, h, w, 3
    torch.save({'frames': (frames * 255).round().to(torch.uint8), 'feats_gt': feats, 'feats_pred': preds,
                'action': actions, 'clips': [val_clips[i] for i in idx]}, os.path.join(out, 'rollout.pt'))

    grid, size = (data_cfg['image_size'] // encoder.patch_size,) * 2, data_cfg['image_size']
    pca = fit_pca(torch.cat([feats.flatten(0, 2), preds.flatten(0, 2)]))
    pca_gt = pca_rgb(feats.flatten(0, 1), pca, grid, size).unflatten(0, feats.shape[:2])  # bs, H+1, h, w, 3
    pca_pred = pca_rgb(preds.flatten(0, 1), pca, grid, size).unflatten(0, preds.shape[:2])  # bs, H, h, w, 3

    plot_grid(torch.stack([frames[:, 0], frames[:, 1], pca_gt[:, 0], pca_gt[:, 1], pca_pred[:, 0]], dim=1),
              os.path.join(out, 'rollout.png'),
              col_titles=['input x_{n-1}', 'GT x_n', 'PCA input', 'PCA GT', 'PCA pred'])
    if H > 1:
        blank = torch.ones_like(pca_pred[:, :1])  # no prediction for the first (given) frame
        for i in range(len(idx)):
            plot_grid(torch.stack([frames[i], pca_gt[i], torch.cat([blank, pca_pred], dim=1)[i]]),
                      os.path.join(out, f'sample_{i}.png'), row_titles=['GT', 'PCA GT', 'PCA pred'],
                      col_titles=['t=0 (given)'] + [f't={k} cos {cos[i, k - 1]:.3f}' for k in range(1, H + 1)])
    print(f"saved to {out}")


if __name__ == '__main__':
    main()
