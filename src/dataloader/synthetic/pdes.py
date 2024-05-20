import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from matplotlib import pyplot as plt
from scipy.special import erf


class PDEs:
    def __init__(self, dx, dy, bc_mask):
        self.dx, self.dy = dx, dy
        self.bc_mask = bc_mask

        # Convolution kernels for second derivatives
        self.kernel_dx2 = np.array([[1, -2, 1]]) / self.dx ** 2
        self.kernel_dy2 = np.array([[1], [-2], [1]]) / self.dy ** 2
        # self.kernel_dxdy = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) / self.dx ** 2
        self.kernel_dxdy = torch.tensor([[0.25, 0.5, 0.25], [0.5, -3, 0.5], [0.25, 0.5, 0.25]]) / self.dx ** 2

        self.kernel_dxdy = self.kernel_dxdy.float().unsqueeze(0).unsqueeze(0)

        self.set_params()

    def set_params(self):

        self.C =  np.random.rand() * 10 * 0 + 0

    def wave_equation(self, u, dudt):
        """
        Returns the time derivative of the input field according to the wave equation. With nonlinear dampening
        """
        dampening = 0.1

        d2u_dx2 = self._laplacian(u)  # convolve2d(u, self.kernel_dxdy, mode='same', boundary='symm')

        d2udt2 = 0.1 * d2u_dx2 - dampening * dudt
        d2udt2 = self._sigmoid_clamp(d2udt2, C=self.C)

        # Boundary Conditions
        dudt[self.bc_mask] = 0
        d2udt2[self.bc_mask] = 0

        # plt.imshow(u)
        # plt.show()
        # plt.imshow(d2u_dx2)
        # plt.show()

        return dudt, d2udt2

    def _laplacian(self, u):
        """
        Returns the Laplacian of the input field.
        u.shape = (H, W)
        """
        u = u.unsqueeze(0).unsqueeze(0)
        u = F.pad(u, pad=(1, 1, 1, 1), mode='reflect')
        result = F.conv2d(u, self.kernel_dxdy)

        return result.squeeze()

    def _sigmoid_clamp(self, x, C=1.0):
        x = C * erf(x / C)
        # sigmoid = torch.sigmoid(x / C) - 0.5
        # transformed_output = sigmoid * C * 4
        return x
