import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class IQFile:
    """
    Handles reading raw .iqdata files.
    Assumes float32 interleaved I/Q.
    """

    def __init__(self, path, dtype=np.float32):
        self.path = Path(path)
        self.dtype = dtype

    def load(self):
        raw = np.fromfile(self.path, dtype=self.dtype)

        # reshape to [time, 2]
        iq = raw.reshape(-1, 2)

        # convert to torch tensor
        iq = torch.from_numpy(iq).float()

        # transpose to [2, time]
        return iq.T
    

class IQSeparationDataset(Dataset):
    """
    Directory structure example:

    root/
        sample_001/
            mixture.iqdata
            src0.iqdata
            src1.iqdata
        sample_002/
            ...
    """

    def __init__(self, root, segment_length=None, num_sources=2):
        self.root = Path(root)
        self.segment_length = segment_length
        self.num_sources = num_sources

        self.examples = sorted(self.root.glob("*"))

    def _load_iq(self, path):
        raw = np.fromfile(path, dtype=np.float32)
        iq = raw.reshape(-1, 2)
        iq = torch.from_numpy(iq).float()
        return iq.T  # (2, T)

    def _random_segment(self, x):
        if self.segment_length is None:
            return x

        T = x.shape[-1]
        if T <= self.segment_length:
            return x

        start = torch.randint(0, T - self.segment_length, (1,)).item()
        return x[:, start:start+self.segment_length]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        folder = self.examples[idx]

        mixture = self._load_iq(folder / "mixture.iqdata")

        sources = []
        for i in range(self.num_sources):
            src = self._load_iq(folder / f"src{i}.iqdata")
            sources.append(src)

        sources = torch.stack(sources, dim=0)  # (N, 2, T)

        mixture = self._random_segment(mixture)
        sources = self._random_segment(sources.view(-1, sources.shape[-1]))
        sources = sources.view(self.num_sources, 2, -1)

        return mixture, sources