import numpy as np

from config import BOXES_DTYPE


def balance_undersample(
    data: list[tuple], rng: (np.random.Generator | None) = None
) -> list[tuple]:
    """
    for a list of event samples and their boxes, determine
    the class with the largest number of samples and
    randomly prune the samples of this class so that its
    number equals the next higher number of samples appearing
    in the rest of the classes w.r.t the median of the number
    of samples in the rest of the classes.
    """

    if rng is None:
        rng = np.random.default_rng()

    assert len(data) > 0

    boxes = np.array([bx for evt, bx in data], dtype=BOXES_DTYPE)
    events = [evt for evt, bx in data]

    class_count: dict[int, int] = {}

    for evt, box in data:
        cl_id = box["class_id"]
        class_count[cl_id] = class_count.get(cl_id, 0) + 1

    # don't do anything if there is only one class in the data
    if len(class_count) == 1:
        return data

    cl_id_max = list(class_count.keys())[int(np.argmax(list(class_count.values())))]
    num_samples_median_rest = int(
        np.percentile(
            [v for k, v in class_count.items() if k != cl_id_max], 50, method="higher"
        )
    )

    idx_cl_id = np.where(boxes["class_id"] == cl_id_max)[0]
    rng.shuffle(idx_cl_id)
    idx_cl_id_exclude = idx_cl_id[:-num_samples_median_rest]

    boxes_new = np.array(
        [bx for i, bx in enumerate(boxes) if i not in idx_cl_id_exclude],
        dtype=BOXES_DTYPE,
    )

    events_new = [
        evt.copy() for i, evt in enumerate(events) if i not in idx_cl_id_exclude
    ]

    return list(zip(events_new, boxes_new))


def balance_oversample(
    data: list[tuple], rng: (np.random.Generator | None) = None
) -> list[tuple]:
    """
    for a list of event samples and their boxes, determine
    the class with the smallest number of samples and
    randomly repeat the samples of this class so that its
    number equals the next smallest number of samples appearing
    in the rest of the classes w.r.t the median of the number
    of samples in the rest of the classes.
    """

    if rng is None:
        rng = np.random.default_rng()

    n_data = len(data)

    assert n_data > 0

    boxes = np.array([bx for evt, bx in data], dtype=BOXES_DTYPE)
    events = [evt for evt, bx in data]

    class_count: dict[int, int] = {}

    for evt, box in data:
        cl_id = box["class_id"]
        class_count[cl_id] = class_count.get(cl_id, 0) + 1

    # don't do anything if there is only one class in the data
    if len(class_count) == 1:
        return data

    cl_id_min = list(class_count.keys())[int(np.argmin(list(class_count.values())))]
    num_samples_median_rest = int(
        np.percentile(
            [v for k, v in class_count.items() if k != cl_id_min], 50, method="lower"
        )
    )

    idx_cl_id = np.where(boxes["class_id"] == cl_id_min)[0]

    num_new_required = num_samples_median_rest - len(idx_cl_id)
    # we need at least this many repetitions to get num_new_required samples
    num_repeat = int(num_new_required // len(idx_cl_id)) + 1

    # repeat, shuffle and then take the first num_new_required elements.
    idx_cl_id = np.repeat(idx_cl_id, num_repeat)
    rng.shuffle(idx_cl_id)
    idx_cl_id = idx_cl_id[:num_new_required]

    idx_total_new = np.append(np.arange(n_data), idx_cl_id)

    boxes_new = np.array(
        [boxes[i] for i in idx_total_new],
        dtype=BOXES_DTYPE,
    )

    events_new = [events[i].copy() for i in idx_total_new]

    return list(zip(events_new, boxes_new))


def lim_samples_rnd(
    data: list[tuple], n_max: int, rng: (np.random.Generator | None) = None
) -> list[tuple]:
    """
    Randomly prune samples from data if n_max is smaller than the
    number of samples in data. This does not factor in classes, so
    you should probably balance the data before pruning.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_data = len(data)

    if n_max >= n_data:
        return data

    n_prune = n_data - n_max

    idx_prune = np.arange(n_data)
    rng.shuffle(idx_prune)
    idx_prune = idx_prune[:n_prune]

    data_prune = [
        (evt.copy(), bx.copy())
        for i, (evt, bx) in enumerate(data)
        if i not in idx_prune
    ]

    return data_prune


if __name__ == "__main__":
    folder = ["/its/home/fs388/repos/lgmd-automotive/data/train_a/"]
    print("loading data")
    data = load_data(folder)
    print("balancing data")
    data_bal_prune = lim_samples_rnd(
        balance_oversample(balance_undersample(data)), 10000
    )

    __import__("ipdb").set_trace()
