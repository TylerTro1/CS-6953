# Provided demucs/iq.py

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class IQDataset(Dataset):
    def __init__(self, root, sources, segment=None):
        self.root = Path(root)
        self.sources = sources
        self.segment = segment # Boolean for self-segmentation
        self.examples = list(self.root.glob("*.iqdata"))

    def load_iq(self, path):
        raw = np.fromfile(path, dtype=np.float32)
        iq = raw.reshape(-1, 2)  # [time, 2]
        return iq.T  # [2, time]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        path = self.examples[idx]

        iq = self.load_iq(path)
        iq = torch.from_numpy(iq)

        if self.segment:
            T = iq.shape[-1]
            if T > self.segment:
                start = torch.randint(0, T - self.segment, (1,))
                iq = iq[:, start:start + self.segment]

        return iq