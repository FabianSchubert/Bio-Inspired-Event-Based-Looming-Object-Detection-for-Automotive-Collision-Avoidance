#! /usr/bin/env python3

import multiprocessing as mp
import time

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)


class ImagePlotter:
    def __init__(self, shape):
        self.shape = shape
        self.dat = np.zeros(shape)

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

        self.img = self.ax.imshow(self.dat, vmin=-0.01, vmax=0.01)

        self.ax.invert_yaxis()

        timer = self.fig.canvas.new_timer(interval=10)
        timer.add_callback(self.call_back)
        timer.start()

        print("...done")
        plt.show()


class ProcessPlotter:
    def __init__(self, n_buff=200):
        self.n_buff = n_buff
        self.y = np.zeros((n_buff, 2))

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
        self.ln1.set_ydata(self.y[:, 0])
        self.ln2.set_ydata(self.y[:, 1])
        self.fig.canvas.draw()
        return True

    def __call__(self, pipe):
        print("starting plotter...")

        self.pipe = pipe
        self.fig, self.ax = plt.subplots()

        (self.ln1, self.ln2) = self.ax.plot(self.y, "-o", markersize=3)

        self.ax.set_ylim([-0.01, 1.0])

        timer = self.fig.canvas.new_timer(interval=10)
        timer.add_callback(self.call_back)
        timer.start()

        print("...done")
        plt.show()


class NBPlot:
    def __init__(self, mode="line", **args):
        self.plot_pipe, plotter_pipe = mp.Pipe()
        if mode == "line":
            self.plotter = ProcessPlotter()
        elif mode == "image":
            self.plotter = ImagePlotter(shape=args["shape"])
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