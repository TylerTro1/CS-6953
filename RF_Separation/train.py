from torch.utils.data import DataLoader
from RFSepNet import RFSepNet

dataset = IQSeparationDataset(
    root="-------",
    segment_length=65536,
    num_sources=2
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4
)


def __getitem__(self, idx):
    folder = self.examples[idx]

    sources = []
    for i in range(self.num_sources):
        src = self._load_iq(folder / f"src{i}.iqdata")
        src = self._random_segment(src)
        sources.append(src)

    sources = torch.stack(sources, dim=0)

    mixture = torch.sum(sources, dim=0)

    return mixture, sources