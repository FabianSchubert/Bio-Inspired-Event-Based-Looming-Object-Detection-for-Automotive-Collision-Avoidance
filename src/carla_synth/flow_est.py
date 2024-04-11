import numpy as np
from scipy.signal import convolve2d


class NumpyDelayArray:

    def __init__(self, shape: tuple, n_delay: int) -> None:
        self.n_delay = n_delay
        self._data = np.zeros((n_delay + 1,) + shape)
        self.idx = 0

    @property
    def new_data(self) -> np.ndarray:
        return self._data[(self.idx - 1) % (self.n_delay + 1)]

    @new_data.setter
    def new_data(self, arr: np.ndarray) -> None:
        self._data[self.idx] = arr
        self.idx = (self.idx + 1) % (self.n_delay + 1)

    @property
    def out_data(self) -> np.ndarray:
        return self._data[self.idx]


class FlowEst:

    def __init__(self, params: dict) -> None:
        self.input_width, self.input_height = (
            params["INPUT_WIDTH"],
            params["INPUT_HEIGHT"],
        )

        self.kernel_width, self.kernel_height = (
            params["KERNEL_WIDTH"],
            params["KERNEL_HEIGHT"],
        )

        self.s_width = self.input_width - self.kernel_width + 1
        self.s_height = self.input_height - self.kernel_height + 1

        self.kernel_half_width = (self.kernel_width - 1) // 2
        self.kernel_half_height = (self.kernel_height - 1) // 2

        self.kernel_g = params["KERNEL_G"]
        self.kernel_d = params["KERNEL_D"]

        self.tau_syn_in_s_i = params["TAU_SYN_IN_S_I"]
        self.tau_syn_in_s_e = params["TAU_SYN_IN_S_E"]

        self.w_in_s_e = params["W_IN_S_E"]

        self.tau_mem_p = params["TAU_MEM_P"]
        self.v_thresh_p = params["V_THRESH_P"]
        self.v_reset_p = params["V_RESET_P"]
        self.curr_evt_p = params["INPUT_EVENT_CURRENT"]

        self.tau_mem_s = params["TAU_MEM_S"]
        self.v_thresh_s = params["V_THRESH_S"]
        self.v_reset_s = params["V_RESET_S"]

        self.dt = params["DT_MS"]

        # internals

        self.v_p = np.zeros((self.input_height, self.input_width))
        self.spike_p = np.zeros((self.input_height, self.input_width), dtype=np.uint8)

        self.v_s = np.zeros((self.s_height, self.s_width))
        self.spike_s = np.zeros((self.s_height, self.s_width), dtype=np.uint8)

        self.i_in_s_del = NumpyDelayArray(self.v_s.shape, self.kernel_d)
        self.i_in_s = np.zeros(self.v_s.shape)
        # self.i_in_e = np.zeros(self.v_s.shape)

        self.x, self.y = np.meshgrid(
            np.arange(self.s_width) - self.s_width // 2,
            np.arange(self.s_height) - self.s_height // 2,
        )

        norm = np.sqrt(self.x**2.0 + self.y**2.0) + 1e2

        self.xnorm, self.ynorm = self.x / (norm**2.0), self.y / (norm**2.0)

    def step(self, i_in: np.ndarray) -> None:

        self.spike_p[:] = np.maximum(0.0, i_in)
        #self.spike_p[:] = i_in

        self.i_in_s[:] = -convolve2d(self.spike_p, self.kernel_g, mode="valid")

        self.i_in_s_del.new_data = self.i_in_s[:]

        dx = self.i_in_s[1:-1, 2:] - self.i_in_s[1:-1, :-2]
        dy = self.i_in_s[2:, 1:-1] - self.i_in_s[:-2, 1:-1]
        dt = self.i_in_s[1:-1, 1:-1] - self.i_in_s_del.out_data[1:-1, 1:-1]

        vx = -np.sign(dx) * dt / (np.abs(dx) + 1e-0)
        vy = -np.sign(dy) * dt / (np.abs(dy) + 1e-0)  # -dt / (dy + 1e-3)

        vx -= 0.0 * vx.mean()
        vy -= 0.0 * vy.mean()

        self.v_s[1:-1, 1:-1] += (
            self.dt
            * (
                vx * self.xnorm[1:-1, 1:-1]
                + vy * self.ynorm[1:-1, 1:-1]
                - self.v_s[1:-1, 1:-1]
            )
            / self.tau_mem_s
        )

        # self.v_s[1:-1, 1:-1] = -dt * np.sign(dx) / (np.abs(dx) + 1e-0)

        # self.v_s[1:-1, 1:-1] = vx  # self.i_in_s
        # self.v_s[:] = self.i_in_s

        # self.v_s += (
        #    self.dt
        #    * (self.i_in_s * 0.0 + self.i_in_e * np.exp(self.i_in_s * 20.0) - self.v_s)
        #    / self.tau_mem_s
        # )

        # self.spike_s[:] = 1 * (self.v_s >= self.v_thresh_s)
        # self.v_s -= self.spike_s * self.v_thresh_s

        # self.v_p += self.dt * (i_in * self.curr_evt_p - self.v_p) / self.tau_mem_p
        # self.spike_p[:] = self.v_p >= self.v_thresh_p
        # self.v_p -= self.spike_p * self.v_thresh_p

        # self.i_in_e[:] += (
        #    self.dt
        #    * (
        #        (
        #            self.spike_p[
        #                self.kernel_half_height : -self.kernel_half_height,
        #                self.kernel_half_width : -self.kernel_half_width,
        #            ]
        #            * self.w_in_s_e
        #        )
        #        - self.i_in_e
        #    )
        #    / self.tau_syn_in_s_e
        # )

        # self.i_in_s[:] += (
        #    self.dt * (self.i_in_s_del.out_data - self.i_in_s) / self.tau_syn_in_s_i
        # )