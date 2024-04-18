import os

import numpy as np

from glob import glob

import torch

from torch.utils.data.sampler import WeightedRandomSampler

from typing import Union


def list_collate_fn(list_items):
    x = []
    y = []
    for x_, y_ in list_items:
        x.append(x_)
        y.append(y_)
    return x, y


class EventDataSet:
    def __init__(
        self,
        folders: list,
        event_cache_size: int = 1000,
        file_cache_size: int = 20,
        max_samples_per_class: Union[int, None] = None,
        print_retr_items: bool = False,
    ):
        self.folders = folders

        self.retr_items = 0
        self.print_retr_items = print_retr_items

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

        self.event_cache = {}
        self.event_cache_size = event_cache_size
        self.getitem_hist = []

        self.file_cache = {}
        self.file_cache_size = file_cache_size
        self.file_cache_hist = []

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

        self.samples_labels, self.unique_labels, self.label_count = self.get_labels()

        if max_samples_per_class is not None:
            # limit the number of samples per class
            idxs_to_remove = np.zeros(self.num_samples, dtype=bool)
            for label in self.unique_labels:
                print("Label", label)
                idxs = np.where(self.samples_labels == label)[0]
                if len(idxs) > max_samples_per_class:
                    np.random.shuffle(idxs)
                    idxs_to_remove[idxs[max_samples_per_class:]] = True

        # filter out the samples that are to be removed
        _new_items = [
            v
            for k, v in self.idx_to_files_boxes_box_idx.items()
            if not idxs_to_remove[k]
        ]

        # create new "hash map"
        self.idx_to_files_boxes_box_idx = dict(zip(range(len(_new_items)), _new_items))

        # update the samples_labels, unique_labels, label_count
        self.samples_labels, self.unique_labels, self.label_count = self.get_labels()

        # create a dictionary with the weights for each label (inverses of the label count)
        self.unique_labels_weight = dict(
            zip(self.unique_labels, self.num_samples / self.label_count)
        )

        # create a list with the weights for each sample
        self.samples_labels_weight = np.array(
            [self.unique_labels_weight[label] for label in self.samples_labels]
        )

        self.samples_labels_weight = torch.from_numpy(self.samples_labels_weight)
        self.samples_labels_weight = self.samples_labels_weight.double()
        self.balanced_sampler = WeightedRandomSampler(
            self.samples_labels_weight, self.num_samples
        )

    @property
    def num_samples(self):
        return len(self.idx_to_files_boxes_box_idx)

    def get_labels(self):
        samples_labels = []
        for i in range(self.num_samples):
            fl, box, boxidx = self.idx_to_files_boxes_box_idx[i]
            samples_labels.append(box[5])
        samples_labels = np.array(samples_labels).astype(int)

        unique_labels, label_count = np.unique(samples_labels, return_counts=True)

        return samples_labels, unique_labels, label_count

    def __getitem__(self, index):
        self.retr_items += 1
        if self.print_retr_items:
            print("Retrieved items", self.retr_items)
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

        if fl in self.file_cache:
            # print("File in cache")
            eventfl = self.file_cache[fl]
        else:
            if len(self.file_cache) == self.file_cache_size:
                self.file_cache.pop(self.file_cache_hist.pop(0))
            eventfl = np.load(fl + "_td.npy", allow_pickle=True)
            self.file_cache[fl] = eventfl
            self.file_cache_hist.append(fl)

        events = eventfl[boxidx]

        return events, box[5]

    def __len__(self):
        return self.num_samples


if __name__ == "__main__":
    dataset = EventDataSet(
        ["../../data/box_events/train_a/"], max_samples_per_class=9000
    )
    print(len(dataset))
    print(dataset.label_count)
    # print(dataset[0])
