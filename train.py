""" Joint training of the IDM and the flow-matching world model on frozen DINOv3 features (single GPU).

    python train.py configs/training_toy_data.yaml
"""
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.data import UniformMotionDataset, list_clips
from src.idm import IDM
from src.model import LatentActionWorldModel
from src.transformer_block import Encoder
from src.world_model import WorldModel


@torch.no_grad()
def encode_pairs(encoder, images):
    """ images: bs, 2, 3, H, W -> bs, 2, N, C float32 backbone features """
    with torch.autocast('cuda', dtype=torch.bfloat16):
        feats = encoder(images.flatten(0, 1))  # 2 bs, N, C
    return feats.float().unflatten(0, (images.shape[0], 2))


@torch.no_grad()
def estimate_feature_stats(encoder, loader, num_batches):
    """ Per-channel mean / std of the backbone features over the first num_batches batches """
    total, total_sq, count = 0.0, 0.0, 0
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        feats = encode_pairs(encoder, batch['frames'].cuda(non_blocking=True)).flatten(0, 2)  # tokens, C
        total = total + feats.sum(0)
        total_sq = total_sq + feats.pow(2).sum(0)
        count += feats.shape[0]
    mean = total / count
    return mean, (total_sq / count - mean.pow(2)).clamp_min(0).sqrt()


@torch.no_grad()
def evaluate(model, encoder, loader, num_batches, sample_steps):
    """ Metrics on held-out clips:
      loss_flow        flow-matching loss with the IDM's action (one noise draw)
      cos_pred         sampled f_n vs true f_n (per-token cosine), with the IDM's action
      cos_shuffled     same with actions shuffled across the batch; close to cos_pred -> the action is ignored
      cos_copy         baseline f_n = f_{n-1} """
    model.eval()
    sums, n = {}, 0
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        pair = encode_pairs(encoder, batch['frames'].cuda(non_blocking=True))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            losses = model(pair)
            action = model.infer_action(pair)
            context, target = pair[:, 0], pair[:, 1]
            pred = model.predict(context, action, sample_steps).float()
            pred_shuffled = model.predict(context, action[torch.randperm(len(action))], sample_steps).float()

        metrics = {
            'loss_flow': losses['loss_flow'],
            'cos_pred': F.cosine_similarity(pred, target, dim=-1).mean(),
            'cos_shuffled': F.cosine_similarity(pred_shuffled, target, dim=-1).mean(),
            'cos_copy': F.cosine_similarity(context, target, dim=-1).mean(),
        }
        for k, v in metrics.items():
            sums[k] = sums.get(k, 0.0) + v.item()
        n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in sums.items()}


def lr_at(step, cfg):
    """ Linear warmup, then cosine decay to 10% of the peak """
    if step < cfg['warmup_steps']:
        return cfg['lr'] * (step + 1) / cfg['warmup_steps']
    progress = min((step - cfg['warmup_steps']) / max(cfg['max_steps'] - cfg['warmup_steps'], 1), 1.0)
    return cfg['lr'] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def main():
    with open(sys.argv[1] if len(sys.argv) > 1 else 'configs/training_toy_data.yaml') as f:
        cfg = yaml.safe_load(f)
    data_cfg, train_cfg, log_cfg = cfg['data'], cfg['train'], cfg['log']
    torch.manual_seed(train_cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(log_cfg['out_dir'], exist_ok=True)
    with open(os.path.join(log_cfg['out_dir'], 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # ---- data ----
    clips = list_clips(data_cfg['path'])
    train_clips, val_clips = clips[:-data_cfg['val_clips']], clips[-data_cfg['val_clips']:]
    train_set = UniformMotionDataset(data_cfg['path'], train_clips, data_cfg['frame_gap'], data_cfg['image_size'])
    val_set = UniformMotionDataset(data_cfg['path'], val_clips, data_cfg['frame_gap'], data_cfg['image_size'], random_start=False)
    train_loader = DataLoader(train_set, batch_size=train_cfg['batch_size'], shuffle=True, drop_last=True,
                              num_workers=data_cfg['num_workers'], pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=train_cfg['batch_size'], num_workers=data_cfg['num_workers'])
    print(f"clips: {len(train_set)} train / {len(val_set)} val")

    # ---- models ----
    encoder = Encoder(cfg['backbone']['name'], layers=tuple(cfg['backbone']['layers'])).cuda()
    grid = (data_cfg['image_size'] // encoder.patch_size,) * 2
    idm_cfg, wm_cfg = cfg['idm'], cfg['world_model']
    idm = IDM(input_dim=encoder.dim, emb_dim=idm_cfg['emb_dim'], num_layers=idm_cfg['num_layers'],
              num_heads=idm_cfg['num_heads'], ffn_ratio=idm_cfg['ffn_ratio'], latent_dim=idm_cfg['latent_dim'],
              grid_size=grid, num_queries=idm_cfg['num_queries'])
    wm = WorldModel(in_dim=encoder.dim, action_dim=idm_cfg['latent_dim'], grid_size=grid,
                    hidden_size=tuple(wm_cfg['hidden_size']), depth=tuple(wm_cfg['depth']),
                    num_heads=tuple(wm_cfg['num_heads']), mlp_ratio=wm_cfg['mlp_ratio'])
    model = LatentActionWorldModel(idm, wm, time_shift=cfg['flow']['time_shift'], kl_weight=idm_cfg['kl_weight']).cuda()
    print(f"params: IDM {sum(p.numel() for p in idm.parameters()) / 1e6:.1f}M | "
          f"world model {sum(p.numel() for p in wm.parameters()) / 1e6:.1f}M | features {grid} x {encoder.dim}")

    # no weight decay on biases, norms and other 1-D parameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{'params': [p for p in params if p.dim() >= 2], 'weight_decay': train_cfg['weight_decay']},
                                   {'params': [p for p in params if p.dim() < 2], 'weight_decay': 0.0}],
                                  lr=train_cfg['lr'], betas=(0.9, 0.95))

    step = 0
    if log_cfg['resume']:
        ckpt = torch.load(log_cfg['resume'], map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        step = ckpt['step']
        print(f"resumed from {log_cfg['resume']} at step {step}")
    else:
        mean, std = estimate_feature_stats(encoder, train_loader, train_cfg['stats_batches'])
        model.set_feature_stats(mean, std)
        print(f"feature stats: mean |mu| {mean.abs().mean():.3f}, mean std {std.mean():.3f}")

    def save(name):
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step, 'config': cfg},
                   os.path.join(log_cfg['out_dir'], name))

    # ---- train ----
    log_file = open(os.path.join(log_cfg['out_dir'], 'log.jsonl'), 'a')
    model.train()
    t0 = time.time()
    while step < train_cfg['max_steps']:
        for batch in train_loader:
            if step >= train_cfg['max_steps']:
                break
            for g in optimizer.param_groups:
                g['lr'] = lr_at(step, train_cfg)

            pair = encode_pairs(encoder, batch['frames'].cuda(non_blocking=True))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                losses = model(pair)
            optimizer.zero_grad(set_to_none=True)
            losses['loss'].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['grad_clip'])
            optimizer.step()
            step += 1

            if step % log_cfg['log_every'] == 0:
                stats = {k: v.item() for k, v in losses.items()}
                stats.update(step=step, lr=optimizer.param_groups[0]['lr'], grad_norm=grad_norm.item(),
                             sec_per_step=(time.time() - t0) / log_cfg['log_every'])
                t0 = time.time()
                print(' | '.join(f"{k} {v:.4g}" for k, v in stats.items()), flush=True)
                log_file.write(json.dumps(stats) + '\n')
                log_file.flush()

            if step % log_cfg['eval_every'] == 0:
                metrics = evaluate(model, encoder, val_loader, log_cfg['eval_batches'], cfg['flow']['sample_steps'])
                print('eval | ' + ' | '.join(f"{k} {v:.4f}" for k, v in metrics.items()), flush=True)
                log_file.write(json.dumps({'step': step, **{f'eval/{k}': v for k, v in metrics.items()}}) + '\n')
                log_file.flush()

            if step % log_cfg['save_every'] == 0:
                save(f'step_{step}.pt')
                save('last.pt')
    save('last.pt')


if __name__ == '__main__':
    main()
