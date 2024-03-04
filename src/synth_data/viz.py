import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from typing import Union


def gen_evt_hist(evts: np.ndarray, t: float, dt: float, w: int, h: int) -> np.ndarray:
    evts_filt = evts[(evts["t"] >= t) & (evts["t"] < (t + dt))]

    hist = np.histogram2d(
        evts_filt["x"],
        evts_filt["y"],
        bins=[np.arange(w + 1), np.arange(h + 1)],
        weights=evts_filt["p"].astype("i1") * 2 - 1,
    )

    img = np.minimum(1.0, np.maximum(-1.0, hist[0].T))

    return img


def plot_event_slice(
    evts: np.ndarray, t: float, dt: float, w: int, h: int, ax: Union[None, Axes] = None
) -> AxesImage:

    img = gen_evt_hist(evts, t, dt, w, h)

    if not ax:
        fig, ax = plt.subplots()

    aximg = ax.imshow(img, vmin=-1.0, vmax=1.0, cmap="gray")

    if not ax:
        fig.tight_layout()
        plt.show()

    return aximg


def play_event_anim(
    evts: np.ndarray, t_start: float, t_end: float, dt: float, w: int, h: int
) -> None:
    fig, ax = plt.subplots()

    t_run = t_end - t_start
    n_frames = int(t_run / dt)

    t_frames = t_start + np.arange(n_frames) * dt

    aximg = plot_event_slice(evts, t_start, dt, w, h, ax)

    def init():
        return (aximg,)

    def update(frame):
        aximg.set_data(gen_evt_hist(evts, frame, dt, w, h))
        return (aximg,)

    ani = FuncAnimation(
        fig, update, frames=t_frames, init_func=init, blit=True, interval=dt
    )

    plt.show()
