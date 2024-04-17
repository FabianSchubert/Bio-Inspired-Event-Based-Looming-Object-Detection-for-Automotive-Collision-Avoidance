import os

import numpy as np

from glob import glob

import torch

from torch.utils.data.sampler import WeightedRandomSampler

def list_collate_fn(list_items):
     x = []
     y = []
     for x_, y_ in list_items:
         x.append(x_)
         y.append(y_)
     return x, y

class EventDataSet:
    def __init__(self, folders: list, event_cache_size: int = 1000):
        self.folders = folders

        self.folders_to_files = {}
        self.files_to_boxes = {}
        self.idx_to_files_boxes_box_idx = {}

        i = 0
        for folder in folders:
            files = glob(os.path.join(folder, "*td.npy"))
            files = [fl.split("_td.npy")[0] for fl in files]
            self.folders_to_files[folder] = files
            for fl in files:
                boxes = np.load(fl + "_bbox.npy", allow_pickle=True)
                self.files_to_boxes[fl] = boxes
                for j, box in enumerate(boxes):
                    self.idx_to_files_boxes_box_idx[i] = (fl, box, j)
                    i += 1

        self.num_samples = len(self.idx_to_files_boxes_box_idx)

        self.event_cache = {}
        self.event_cache_size = event_cache_size
        self.getitem_hist = []

        # create a weighted random sampler for balanced training
        # the sampler needs to be passed to a dataloader.

        # go through all samples and store the label found in box[5]
        self.samples_labels = []
        for i in range(self.num_samples):
            fl, box, boxidx = self.idx_to_files_boxes_box_idx[i]
            self.samples_labels.append(box[5])
        self.samples_labels = np.array(self.samples_labels).astype(int)

        self.unique_labels, self.label_count = np.unique(
            self.samples_labels, return_counts=True
        )
        self.unique_labels_weight = dict(zip(self.unique_labels, 1 / self.label_count))

        self.samples_labels_weight = np.array(
            [self.unique_labels_weight[label] for label in self.samples_labels]
        )

        self.samples_labels_weight = torch.from_numpy(self.samples_labels_weight)
        self.samples_labels_weight = self.samples_labels_weight.double()
        self.balanced_sampler = WeightedRandomSampler(self.samples_labels_weight, self.num_samples)

    def __getitem__(self, index):
        # cache a limited number of samples
        if index in self.event_cache:
            return self.event_cache[index]
        else:
            if len(self.event_cache) == self.event_cache_size:
                self.event_cache.pop(self.getitem_hist.pop(0))
            events, label = self.load_file(index)
            self.event_cache[index] = (events, label)
            self.getitem_hist.append(index)
            return events, label

    def load_file(self, index):
        fl, box, boxidx = self.idx_to_files_boxes_box_idx[index]
        events = np.load(fl + "_td.npy", allow_pickle=True)[boxidx]
        return events, box[5]

    def __len__(self):
        return self.num_samples
