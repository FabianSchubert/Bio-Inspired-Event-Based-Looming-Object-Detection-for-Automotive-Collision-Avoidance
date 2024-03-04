import numpy as np


def gabor_filters(
    ny, nx, channels, lmbd: float = 1.5, p: float = 0.0, s: float = 0.6, g: float = 1.0
):
    Y, X = np.meshgrid(
        np.linspace(-1.0, 1.0, ny), np.linspace(-1.0, 1.0, nx), indexing="ij"
    )

    weights = np.empty(
        (ny, nx, 1, channels)
    )  # add dummy dimension for mlgenn weight init consistency (1 input channel)

    for k in range(channels):
        rot = np.pi * 2.0 * k / channels

        e_x = np.array([np.cos(rot), np.sin(rot)])
        e_y = np.array([-e_x[1], e_x[0]])

        _X = e_x[0] * X + e_x[1] * Y
        _Y = e_y[0] * X + e_y[1] * Y

        _gabor = np.exp(
            -(_X**2.0 + g**2.0 * _Y**2.0) / (2.0 * s**2.0)
        ) * np.sin(2.0 * np.pi * _X / lmbd + p)

        weights[:, :, 0, k] = _gabor

    return weights
