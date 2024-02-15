import numpy as np
import functools


class ComposeAugment:
    def __init__(self, *op_seq):
        self.op_seq = op_seq
        self.compfunc = functools.reduce(
            lambda f, g: lambda x: g(f(x)), self.op_seq, lambda x: x
        )

    def __call__(self, events: np.ndarray) -> np.ndarray:
        return self.compfunc(events)


class FlipHorizontal:
    def __init__(self, sensor_size, rng, flp_chance=0.5):
        self.sensor_size = sensor_size
        self.rng = rng
        self.flp_chance = flp_chance

    def __call__(self, events: np.ndarray) -> np.ndarray:
        flipped_events = np.array(events)
        if self.rng.uniform() <= self.flp_chance:
            flipped_events["x"] = self.sensor_size[0] - flipped_events["x"]

        return flipped_events


class Crop:
    def __init__(self, sensor_size, rng, min_scale=0.33, max_scale=1.0):
        self.sensor_size = sensor_size
        self.rng = rng
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, events: np.ndarray) -> np.ndarray:
        scale = self.rng.uniform(self.min_scale, self.max_scale)

        x0 = self.rng.uniform(0.0, self.sensor_size[0] * (1.0 - scale))
        y0 = self.rng.uniform(0.0, self.sensor_size[1] * (1.0 - scale))
        x1 = x0 + scale * self.sensor_size[0]
        y1 = y0 + scale * self.sensor_size[1]

        cropped_events = np.array(
            events[
                np.where(
                    (events["x"] >= x0)
                    & (events["x"] <= x1)
                    & (events["y"] >= y0)
                    & (events["y"] <= y1)
                )[0]
            ]
        )

        cropped_events["x"] = (cropped_events["x"] - x0) / scale
        cropped_events["y"] = (cropped_events["y"] - y0) / scale

        return cropped_events
