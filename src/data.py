""" Toy video dataset: uniform_motion_30K.hdf5.

Layout: 30 shards (groups '00000' ... '00029') of 1000 clips each; per clip
  video_streams/<shard>[i]     mp4 bytes, 32 frames of 256 x 256 RGB at 10 fps (a ball moving at constant velocity)
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


class UniformMotionDataset(Dataset):
    """ Returns {'frames': 2, 3, image_size, image_size}, ImageNet-normalised (x_{n-1}, x_n).
    Training picks a random start frame per call; with random_start=False the start is fixed per clip,
    so validation sees the same pairs every time """
    def __init__(self, path, clips, frame_gap=1, image_size=256, random_start=True):
        self.path = path
        self.clips = clips
        self.frame_gap = frame_gap
        self.image_size = image_size
        self.random_start = random_start
        self.file = None  # opened lazily so each DataLoader worker gets its own handle

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.path, 'r')
        shard, i = self.clips[idx]
        video = decord.VideoReader(io.BytesIO(self.file['video_streams'][shard][i].tobytes()))
        rng = np.random.default_rng(None if self.random_start else idx)
        n = int(rng.integers(len(video) - self.frame_gap))
        frames = torch.from_numpy(video.get_batch([n, n + self.frame_gap]).asnumpy())  # 2, H, W, 3 uint8
        frames = frames.permute(0, 3, 1, 2).float() / 255
        if frames.shape[-1] != self.image_size:
            frames = F.interpolate(frames, size=(self.image_size, self.image_size), mode='bilinear', antialias=True)
        return {'frames': (frames - IMAGENET_MEAN) / IMAGENET_STD}
