import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from typing import Union

import os

plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"


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
    evts: np.ndarray,
    t_start: float,
    t_end: float,
    dt: float,
    w: int,
    h: int,
    save_frames: Union[None, str] = None,
    save_video: Union[None, str] = None,
) -> None:
    fig, ax = plt.subplots()

    t_run = t_end - t_start
    n_frames = int(t_run / dt)

    t_frames = t_start + np.arange(n_frames) * dt

    frames_it = list(zip(t_frames, np.arange(t_frames.shape[0])))

    aximg = plot_event_slice(evts, t_start, dt, w, h, ax)

    if save_frames:
        if not os.path.exists(save_frames):
            os.makedirs(save_frames)

    def init():
        return (aximg,)

    def update(frame):
        frame_t, frame_idx = frame
        aximg.set_data(gen_evt_hist(evts, frame_t, dt, w, h))

        if save_frames:
            fig.savefig(os.path.join(save_frames, f"anim_{frame_idx}.png"))

        return (aximg,)

    ani = FuncAnimation(
        fig, update, frames=frames_it, init_func=init, blit=True, interval=dt
    )

    if save_video:
        FFwriter = FFMpegWriter(fps=int(1e3 / dt))
        ani.save(save_video, writer=FFwriter)

    plt.show()


def play_var_anim(
    var: np.ndarray,
    t_start: float,
    t_end: float,
    dt_rec: float,
    dt_play: float,
    vmin: float,
    vmax: float,
    save_frames: Union[None, str] = None,
    save_video: Union[None, str] = None,
) -> None:
    fig, ax = plt.subplots()

    t_run = t_end - t_start

    n_frames = int(t_run / dt_play)

    t_frames = t_start + np.arange(n_frames) * dt_play

    ind_frames = (t_frames / dt_rec).astype("int")

    frames_it = list(zip(ind_frames, np.arange(ind_frames.shape[0])))

    aximg = ax.imshow(var[0], vmin=vmin, vmax=vmax)

    plt.colorbar(aximg, ax=ax)

    if save_frames:
        if not os.path.exists(save_frames):
            os.makedirs(save_frames)

    def init():
        return (aximg,)

    def update(frame):
        frame_id, frame_idx = frame

        new_img = np.zeros(var.shape[1:])

        if frame_id < var.shape[0]:
            new_img[:] = var[frame_id]

        aximg.set_data(new_img)

        if save_frames:
            fig.savefig(os.path.join(save_frames, f"anim_{frame_idx}.png"))

        return (aximg,)

    ani = FuncAnimation(
        fig, update, frames=frames_it, init_func=init, blit=True, interval=dt_play
    )

    if save_video:
        FFwriter = FFMpegWriter(fps=int(1e3 / dt_play))
        ani.save(save_video, writer=FFwriter)

    plt.show()
