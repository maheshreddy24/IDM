""" Toy video datasets: uniform_motion_30K.hdf5 (one ball at constant velocity) and collision_30K.hdf5 (two colliding balls).

Layout: shards (groups '00000', '00001', ...) of clips; shard sizes may differ (collision: 29 shards, 26066 clips). Per clip
  video_streams/<shard>[i]     mp4 bytes, 32 frames of 256 x 256 RGB at 10 fps
  position_streams/<shard>[i]  32, 2, 2 ball centres per frame in world units, (left, right) ball x (x, y) (collision only)
Each item is one random frame pair (x_{n-1}, x_n) = (frame n, frame n + frame_gap) of one clip.
"""
import io

import decord
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def list_clips(path):
    """ -> list of (shard, index) for every clip in the file """
    with h5py.File(path, 'r') as f:
        return [(shard, i) for shard in sorted(f['video_streams']) for i in range(len(f['video_streams'][shard]))]


class VideoPairDataset(Dataset):
    """ Returns {'frames': num_frames, 3, image_size, image_size}, ImageNet-normalised frames frame_gap apart
    (num_frames=2: the pair (x_{n-1}, x_n); more for multi-step rollouts), plus 'positions': num_frames, 2, 2
    (left, right ball x (x, y)) when the file has them.
    Training picks a random start frame per call; with random_start=False the start is fixed per clip,
    so validation sees the same pairs every time """
    def __init__(self, path, clips, frame_gap=1, image_size=256, random_start=True, num_frames=2):
        self.path = path
        self.clips = clips
        self.frame_gap = frame_gap
        self.image_size = image_size
        self.random_start = random_start
        self.num_frames = num_frames
        self.file = None  # opened lazily so each DataLoader worker gets its own handle

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.path, 'r')
        shard, i = self.clips[idx]
        video = decord.VideoReader(io.BytesIO(self.file['video_streams'][shard][i].tobytes()))
        rng = np.random.default_rng(None if self.random_start else idx)
        span = (self.num_frames - 1) * self.frame_gap
        n = int(rng.integers(len(video) - span))
        frame_idx = list(range(n, n + span + 1, self.frame_gap))
        frames = torch.from_numpy(video.get_batch(frame_idx).asnumpy())  # T, H, W, 3 uint8
        frames = frames.permute(0, 3, 1, 2).float() / 255
        if frames.shape[-1] != self.image_size:
            frames = F.interpolate(frames, size=(self.image_size, self.image_size), mode='bilinear', antialias=True)
        item = {'frames': (frames - IMAGENET_MEAN) / IMAGENET_STD}
        if 'position_streams' in self.file:
            item['positions'] = torch.from_numpy(self.file['position_streams'][shard][i][frame_idx]).float()  # T, 2, 2
        return item
