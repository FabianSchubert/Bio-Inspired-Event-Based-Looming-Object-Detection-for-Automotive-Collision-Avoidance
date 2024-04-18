from src.viz import gen_evt_hist
import numpy as np

import torch
from torch.utils.data import Dataset


def gen_img_arr(
    event_sequence: list,
    width: int,
    height: int,
    t0_microsecs: float = 0.0,
    dt_microsecs: float = 10000.0,
) -> np.ndarray:
    img_arr = np.empty((len(event_sequence), height, width))
    for i, evts in enumerate(event_sequence):
        img_arr[i] = gen_evt_hist(evts, t0_microsecs, dt_microsecs, width, height)

    return img_arr


def gen_label_arr(
    boxes: list,
) -> np.ndarray:
    labels = np.empty((len(boxes)))
    labels[:] = boxes["class_id"]

    return labels


class AtisImageDataset(Dataset):
    def __init__(
        self,
        dataset: str | list,
        width: int,
        height: int,
        t0_microsecs: float = 0.0,
        dt_microsecs: float = 10000.0,
        transform=None,
    ):
        super().__init__()

        if isinstance(dataset, str):
            dataset = np.load(dataset, allow_pickle=True)
            dataset = [dataset[key] for key in dataset.keys()]

        event_sequence = dataset[0]
        self.label_arr = torch.tensor(dataset[1], dtype=torch.int64)

        self.img_arr = gen_img_arr(
            event_sequence, width, height, t0_microsecs, dt_microsecs
        )
        self.img_arr = torch.unsqueeze(torch.tensor(self.img_arr, dtype=torch.float32), 1)

        self.transform = transform

    def __len__(self):
        return len(self.img_arr)
    
    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.img_arr[idx]), self.label_arr[idx]
        return self.img_arr[idx], self.label_arr[idx]
