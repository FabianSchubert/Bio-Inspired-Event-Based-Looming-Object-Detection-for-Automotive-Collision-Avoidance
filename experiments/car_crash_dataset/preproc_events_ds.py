import numpy as np
import os

_fl_path = os.path.dirname(os.path.abspath(__file__))

files_crash = list(
    map(
        lambda fl: os.path.join(
            _fl_path, "../../data/car_crash_dataset/events_crash", fl
        ),
        os.listdir(os.path.join(_fl_path, "../../data/car_crash_dataset/events_crash")),
    )
)
files_normal = list(
    map(
        lambda fl: os.path.join(
            _fl_path, "../../data/car_crash_dataset/events_normal", fl
        ),
        os.listdir(
            os.path.join(_fl_path, "../../data/car_crash_dataset/events_normal")
        ),
    )
)

new_base_fold = os.path.join(_fl_path, "../../data/car_crash_dataset/events")
os.makedirs(new_base_fold)

DT = 6.26

DTYPES_OUT = [("t", np.uint32), ("x", np.uint16), ("y", np.uint16), ("p", np.uint16)]

# determine average event rates

t = 0
n_e = 0

for fl in files_crash:
    events = np.load(fl)
    t += events["t"].max()
    n_e += len(events)

rate_crash = n_e / t

t = 0
n_e = 0

for fl in files_normal:
    events = np.load(fl)
    t += events["t"].max()
    n_e += len(events)

rate_normal = n_e / t

for k, fl in enumerate(files_crash):
    print(f"processing {fl}, {k+1}/{len(files_crash)} in crash...")
    fold = os.path.join(new_base_fold, f"example_{k}")
    os.makedirs(fold)

    events = np.load(fl)

    n_e = len(events)

    # wiggle times
    events["t"] += np.random.rand(n_e) * DT

    '''
    # downsample events if more on average
    if rate_crash > rate_normal:

        ind = np.arange(n_e)
        np.random.shuffle(ind)
        ind = np.sort(ind[: int(n_e * rate_normal / rate_crash)])

        events = events[ind]
    '''
    events = events.astype(DTYPES_OUT)

    # save metadata
    np.savez(
        os.path.join(fold, "sim_data.npz"),
        t_end=events["t"].max(),
        dt=DT,
        coll_type="cars",
        vel=None,
        diameter_object=None,
    )

    # save events
    np.save(os.path.join(fold, "events.npy"), events)

for k, fl in enumerate(files_normal):
    print(f"processing {fl}, {k+1}/{len(files_normal)} in normal...")
    fold = os.path.join(new_base_fold, f"example_{k+len(files_crash)}")
    os.makedirs(fold)

    events = np.load(fl)

    n_e = len(events)

    # wiggle times
    events["t"] += np.random.rand(n_e) * DT

    '''
    # downsample events if more on average
    if rate_normal > rate_crash:

        ind = np.arange(n_e)
        np.random.shuffle(ind)
        ind = np.sort(ind[: int(n_e * rate_normal / rate_crash)])

        events = events[ind]
    '''
    events = events.astype(DTYPES_OUT)

    # save metadata
    np.savez(
        os.path.join(fold, "sim_data.npz"),
        t_end=events["t"].max(),
        dt=DT,
        coll_type="none",
        vel=None,
        diameter_object=None,
    )

    # save events
    np.save(os.path.join(fold, "events.npy"), events)
