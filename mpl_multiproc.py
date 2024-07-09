#! /usr/bin/env python3

import multiprocessing as mp
import time

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)


class ImagePlotter:
    def __init__(self, shape, vmin=-1., vmax=1., flip_y=False):
        self.shape = shape
        self.dat = np.zeros(shape)
        self.vmin = vmin
        self.vmax = vmax
        self.flip_y = flip_y

    def terminate(self):
        plt.close("all")

    def call_back(self):
        while self.pipe.poll():
            command = self.pipe.recv()
            if command is None:
                self.terminate()
                return False
            else:
                self.dat[:] = command

        self.img.set_data(self.dat)
        # self.img.set_clim(vmin=self.dat.min(), vmax=self.dat.max())
        self.fig.canvas.draw()
        return True

    def __call__(self, pipe):
        print("starting plotter...")
        self.pipe = pipe
        self.fig, self.ax = plt.subplots()

        self.img = self.ax.imshow(self.dat, vmin=self.vmin, vmax=self.vmax)

        if self.flip_y:
            self.ax.invert_yaxis()

        timer = self.fig.canvas.new_timer(interval=100)
        timer.add_callback(self.call_back)
        timer.start()

        print("...done")
        plt.show()


class ProcessPlotter:
    def __init__(self, n_buff=200, n_lines=2, vmin=0., vmax=1.):
        self.n_buff = n_buff
        self.y = np.zeros((n_buff, n_lines))
        self.vmin = vmin
        self.vmax = vmax

    def terminate(self):
        plt.close("all")

    def call_back(self):
        while self.pipe.poll():
            command = self.pipe.recv()
            if command is None:
                self.terminate()
                return False
            else:
                self.y[:-1] = self.y[1:]
                self.y[-1] = command
        for i, ln in enumerate(self.lines):
            ln.set_ydata(self.y[:, i])
        self.fig.canvas.draw()
        return True

    def __call__(self, pipe):
        print("starting plotter...")

        self.pipe = pipe
        self.fig, self.ax = plt.subplots()

        self.lines = self.ax.plot(self.y, "-o", markersize=3)
        # (self.ln1, self.ln2) = self.ax.plot(self.y, "-o", markersize=3)

        self.ax.set_ylim([self.vmin, self.vmax])

        timer = self.fig.canvas.new_timer(interval=100)
        timer.add_callback(self.call_back)
        timer.start()

        print("...done")
        plt.show()


class NBPlot:
    def __init__(self, mode="line", **args):
        self.plot_pipe, plotter_pipe = mp.Pipe()
        if mode == "line":
            self.plotter = ProcessPlotter(**args)
        elif mode == "image":
            self.plotter = ImagePlotter(**args)
        else:
            raise ValueError
        self.plot_process = mp.Process(
            target=self.plotter, args=(plotter_pipe,), daemon=True
        )
        self.plot_process.start()

    def plot(self, data, finished=False):
        send = self.plot_pipe.send
        if finished:
            send(None)
        else:
            send(data)


def main():
    pl = NBPlot()
    for _ in range(10):
        pl.plot(np.random.rand())
        time.sleep(0.5)
    pl.plot(None, finished=True)


if __name__ == "__main__":
    if plt.get_backend() == "MacOSX":
        mp.set_start_method("forkserver")
    main()
