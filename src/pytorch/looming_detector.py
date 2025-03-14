import torch
import torch.nn as nn

import numpy as np

from .network_settings import (
    KERN_HALF_SIZE,
    SIGM_KERNEL,
    R_OUT_SIGM_SCALE_INIT,
    R_OUT_SCALE_INIT,
    TRAIN_WEIGHTS,
    USE_DROPOUT,
    TRAIN_KERNELS,
    DT,
    REG,
    RMO_HEAD,
)


class SigmoidScaleBias(nn.Module):
    def __init__(self, dim, scale=True, bias=True, single_param=False):
        super().__init__()
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1 if single_param else dim))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter("bias", None)
        if scale:
            self.scale = nn.Parameter(torch.Tensor(1 if single_param else dim))
        else:
            self.register_parameter("scale", None)

        if self.bias is not None:
            self.bias.data.uniform_(0, 0)

        if self.scale is not None:
            self.scale.data.uniform_(1, 1)

    def forward(self, input):
        output = input
        if self.scale is not None:
            output *= self.scale.unsqueeze(0).expand_as(output)
        if self.bias is not None:
            output -= self.bias.unsqueeze(0).expand_as(output)
        return torch.sigmoid(output)


class LoomingDetector(nn.Module):
    def __init__(
        self,
        width=320,
        height=240,
        kern_half_size=KERN_HALF_SIZE,
        rmo_head=RMO_HEAD,
        sigm_kern=SIGM_KERNEL,
        reg=REG,
        dt=DT,
        r_out_sigm_scale=R_OUT_SIGM_SCALE_INIT,
        r_out_scale=R_OUT_SCALE_INIT,
        train_weights=TRAIN_WEIGHTS,
        use_dropout=USE_DROPOUT,
        train_kernels=TRAIN_KERNELS,
    ):
        super().__init__()

        self.width = width
        self.height = height
        self.kern_half_size = kern_half_size

        self.kern_size = 2 * kern_half_size + 1

        self.sigm_kern = sigm_kern

        self.reg = reg

        self.dt = dt

        self.train_weights = train_weights

        self.rmo_head = rmo_head
        self.use_dropout = use_dropout

        _x, _y = np.meshgrid(
            np.arange(-kern_half_size, kern_half_size + 1),
            np.arange(-kern_half_size, kern_half_size + 1),
        )

        self.kern_x = nn.Conv2d(
            1, 1, self.kern_size, padding=kern_half_size, bias=False
        )

        self.kern_y = nn.Conv2d(
            1, 1, self.kern_size, padding=kern_half_size, bias=False
        )

        self.kern_norm = nn.Conv2d(
            1, 1, self.kern_size, padding=kern_half_size, bias=False
        )

        _gauss = np.exp(-(_x**2 + _y**2) / (2.0 * (sigm_kern * kern_half_size) ** 2))
        _gauss /= _gauss.sum()

        with torch.no_grad():
            self.kern_norm.weight.data = nn.Parameter(
                torch.tensor(_gauss).float().unsqueeze(0).unsqueeze(0),
            )
            self.kern_norm.weight.requires_grad = train_kernels

            self.kern_x.weight.data = nn.Parameter(
                torch.tensor(_x * _gauss).float().unsqueeze(0).unsqueeze(0),
            )
            self.kern_x.weight.requires_grad = train_kernels

            self.kern_y.weight.data = nn.Parameter(
                torch.tensor(_y * _gauss).float().unsqueeze(0).unsqueeze(0),
            )
            self.kern_y.weight.requires_grad = train_kernels

        self.vx = torch.zeros((1, 1, height, width)).float()
        self.vy = torch.zeros((1, 1, height, width)).float()

        _x, _y = np.meshgrid(
            np.linspace(-1, 1, width),
            np.linspace(-height / width, height / width, height),
        )

        with torch.no_grad():
            self.wx = nn.Parameter(torch.tensor(_x).float().unsqueeze(0).unsqueeze(0))
            self.wy = nn.Parameter(torch.tensor(_y).float().unsqueeze(0).unsqueeze(0))

            self.wx.requires_grad = False
            self.wy.requires_grad = False

        self.fwd = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * height, 2, bias=False),
        )

        with torch.no_grad():
            if not self.train_weights:
                w_pos = np.exp(-(_x**2.0) / (2.0 * 0.25**2)) * np.exp(
                    -(_y**2.0) / (2.0 * 0.25**2)
                )
                w_pos_left = np.zeros_like(w_pos)
                w_pos_left[:, : width // 2] = w_pos[:, : width // 2]
                w_pos_right = np.zeros_like(w_pos)
                w_pos_right[:, width // 2 :] = w_pos[:, width // 2 :]

                self.fwd[1].weight.data[0] = nn.Parameter(
                    torch.tensor(w_pos_left.flatten()).float()
                )
                self.fwd[1].weight.data[1] = nn.Parameter(
                    torch.tensor(w_pos_right.flatten()).float()
                )
            else:
                self.fwd[1].weight.data = nn.Parameter(
                    torch.zeros((2, width * height)).float()
                )

            self.fwd[1].weight.requires_grad = self.train_weights

        self.sigm = SigmoidScaleBias(2, single_param=True)

        with torch.no_grad():
            self.sigm.scale[:] = r_out_sigm_scale

        self.out_scale = nn.Parameter(torch.tensor([r_out_scale]).float())

        self.out_th = nn.Parameter(torch.zeros((1)).float())

        if self.use_dropout:
            self.dropout = nn.Dropout(self.use_dropout)

    def forward(self, x):
        x_p = (x > 0).float()
        x_n = (x < 0).float()

        if self.use_dropout:
            x_p = self.dropout(x_p)
            x_n = self.dropout(x_n)

        I_x_p_prev = self.kern_x(x_p[..., 0:1, :, :])
        I_y_p_prev = self.kern_y(x_p[..., 0:1, :, :])
        I_norm_p_prev = self.kern_norm(x_p[..., 0:1, :, :])

        vx_p = (
            -x_p[..., 1:2, :, :]
            * I_x_p_prev
            * self.dt
            / (self.reg + I_norm_p_prev * self.dt**2)
        )
        vy_p = (
            -x_p[..., 1:2, :, :]
            * I_y_p_prev
            * self.dt
            / (self.reg + I_norm_p_prev * self.dt**2)
        )

        I_x_n_prev = self.kern_x(x_n[..., 0:1, :, :])
        I_y_n_prev = self.kern_y(x_n[..., 0:1, :, :])
        I_norm_n_prev = self.kern_norm(x_n[..., 0:1, :, :])

        vx_n = (
            -x_n[..., 1:2, :, :]
            * I_x_n_prev
            * self.dt
            / (self.reg + I_norm_n_prev * self.dt**2)
        )
        vy_n = (
            -x_n[..., 1:2, :, :]
            * I_y_n_prev
            * self.dt
            / (self.reg + I_norm_n_prev * self.dt**2)
        )

        self.vx = vx_p + vx_n
        self.vy = vy_p + vy_n

        v_project = self.vx * self.wx + self.vy * self.wy

        left_right = self.fwd(v_project)

        if self.rmo_head:
            out = (
                self.out_scale
                * torch.min(left_right, dim=1).values
                * torch.prod(self.sigm(left_right), dim=1)
                - self.out_th
            )
        else:
            out = self.out_scale * left_right.mean(dim=1) - self.out_th

        return out
