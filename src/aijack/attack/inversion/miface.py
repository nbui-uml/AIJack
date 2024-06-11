"""MI-Face attack implementation"""

from matplotlib import pyplot as plt

# from scipy.optimize import approx_fprime

import torch
import torch.nn as nn
from torch.autograd import Function

# if torch.cuda.is_available():
#     import cupy as np
# else:
#     import numpy as np

# from aijack.attack import BaseAttacker


def approx_fprime(x0, F, H, device="cpu"):
    """
    Args:
        x0 (tensor): input tensor
        F (function): F(x) that returns a scalar value
        h (float): Basis object for steps
        device (str or torch.device): cpu/gpu/cuda
    
    Returns:
        tensor: linear approximation of F'(x) with one point
    """
    x0.to(device)
    X0 = x0.repeat(x0.shape[1] * x0.shape[2] * x0.shape[3], 1, 1, 1)
    Y0 = F(X0)
    Y1 = F(torch.add(X0, H.vectors, alpha=1))
    return torch.div(torch.add(Y1, Y0, alpha=-1), H.h).reshape(H.shape)

def approx_fprime2(x0, F, H, device="cpu"):
    """
    Args:
        x0 (tensor): input tensor
        F (function): F(x) that returns a scalar value
        H (float): Basis object for steps
        device (str or torch.device): cpu/gpu/cuda
    
    Returns:
        tensor: linear approximation of F'(x) using two point method
    """
    x0.to(device)
    X0 = x0.repeat(x0.shape[1] * x0.shape[2] * x0.shape[3], 1, 1, 1)
    Y0 = F(torch.add(X0, H.vectors, alpha=-0.5))
    Y1 = F(torch.add(X0, H.vectors, alpha= 0.5))
    return torch.div(torch.add(Y1, Y0, alpha=-1), H.h).reshape(H.shape)

class Basis:
    """
    Step kernel object for vectorized gradient approximation of rank-3 tensor to scalar functions.

    Attributes:
        shape (tuple): shape of function input
        h (float): step size
        kernel (tensor): kernel to pass to function
    """
    def __init__(self, shape, h=1.5e-08, device="cpu"):
        """
        Args:
            shape (tuple): shape of function input
            h (float): step size

        Returns:
            Basis
        """
        self.shape = shape
        self.h = h

        c = shape[1]
        h = shape[2]
        w = shape[3]
        H = torch.zeros(shape).repeat(c * h * w, 1, 1, 1)

        b = 0
        # This is awful but at least I only have to do it once?
        for k in range(c):
            for j in range(h):
                for i in range(w):
                    H[b][k][j][i] = h
                    b += 1
        
        self.vectors = H.to(device)

class MI_FACE:
    """Implementation of model inversion attack
    reference: https://dl.acm.org/doi/pdf/10.1145/2810103.2813677

    Attributes:
        target_model: model of the victim
        input_shape: input shapes of taregt model
        auxterm_func (function): the default is constant function
        process_func (function): the default is identity function
    """

    def __init__(
        self,
        target_model,
        input_shape=(1, 1, 64, 64),
        target_label=0,
        lam=0.01,
        num_itr=100,
        beta=None,
        gamma=None,
        auxterm_func=lambda x: 0,
        process_func=lambda x: x,
        apply_softmax=False,
        device="cpu",
        log_interval=1,
        log_show_img=False,
        show_img_func=lambda x: x * 0.5 + 0.5,
        black_box=False,
    ):
        """Inits MI_FACE
        Args:
            target_model: model of the victim
            input_shape: input shapes of taregt model
            target_label (int): taregt label
            lam (float): step size
            num_itr (int): number of iteration
            beta (int): min number of iterations before considering early stoppage
              defaults to num_itr
            gamma (float): ideal score
              defaults to 0.0
            auxterm_func (function): the default is constant function
            process_func (function): the default is identity function
            black_box (bool): perform black_box attack
              defaults to False
        """
        self.target_model = target_model
        self.input_shape = input_shape
        self.target_label = target_label
        self.lam = lam
        self.num_itr = num_itr
        self.beta = beta if beta else num_itr
        self.gamma = gamma if gamma else 0.0

        self.auxterm_func = auxterm_func
        self.process_func = process_func
        self.device = device
        self.log_interval = log_interval
        self.log_show_img = log_show_img
        self.apply_softmax = apply_softmax
        self.show_img_func = show_img_func
        self.black_box = black_box

        self.log_image = []

    def blackbox_attack(self, init_x=None):
        self.target_model = self.target_model.to(self.device)
        log = []
        if init_x is None:
            x = torch.zeros(self.input_shape, requires_grad=True).to(self.device)
        else:
            init_x = init_x
            x = init_x

        best_score = float("inf")
        best_img = None

        # ONE = torch.ones((x.shape[1] * x.shape[2] * x.shape[3])).to(self.device)
        # Ignore auxterm for now
        cost = (
            lambda x: 1
            - self.target_model(x)[:, [self.target_label]]
        )

        for i in range(self.num_itr):
            x = x.detach()
            c = cost(x).item()
            # grad = approx_fprime(x, cost, 1.5e-08, self.device)
            H = Basis(x.shape, device=self.device)
            grad = approx_fprime2(x, cost, H, device=self.device)

            if c < best_score:
                best_img = x

            with torch.no_grad():
                x -= self.lam * grad
                x = self.process_func(x)
            log.append(c)

            if self.log_interval != 0 and i % self.log_interval == 0:
                print(f"epoch {i}: {c}")
                self._show_img(x)

            self.log_image.append(x.clone())

            # Early stoppage based on lack of improvement
            if i >= self.beta and c >= max(log[i - self.beta : i]):
                break

            # Early stoppage from reaching goal
            if c <= self.gamma:
                break

        self._show_img(x)

        return best_img, log

    def attack(
        self,
        init_x=None,
    ):
        """Execute the model inversion attack on the target model.

        Args:

        Returns:
            best_img: inversed image with the best score
            log : list of all recorded scores
        """
        log = []
        if init_x is None:
            x = torch.zeros(self.input_shape, requires_grad=True).to(self.device)
        else:
            init_x = init_x.to(self.device)
            x = init_x

        best_score = float("inf")
        best_img = None

        for i in range(self.num_itr):
            x = x.detach()
            x.requires_grad = True
            pred = self.target_model(x)
            pred = pred.softmax(dim=1) if self.apply_softmax else pred
            target_pred = pred[:, [self.target_label]]
            c = 1 - target_pred + self.auxterm_func(x)

            c.backward()
            grad = x.grad

            if c.item() < best_score:
                best_img = x

            with torch.no_grad():
                x -= self.lam * grad
                x = self.process_func(x)
            log.append(c.item())

            if self.log_interval != 0 and i % self.log_interval == 0:
                print(f"epoch {i}: {c.item()}")
                self._show_img(x)

            self.log_image.append(x.clone())

            # Early stoppage based on lack of improvement
            if i >= self.beta and c.item() >= max(log[i - self.beta : i]):
                break

            # Early stoppage from reaching goal
            if c.item() <= self.gamma:
                break

        self._show_img(x)

        return best_img, log

    def _show_img(self, x):
        if self.log_show_img:
            if self.input_shape[1] == 1:
                plt.imshow(
                    self.show_img_func(x.detach().cpu().numpy()[0][0]),
                    cmap="gray",
                )
                plt.show()
            else:
                plt.imshow(
                    self.show_img_func(x.detach().cpu().numpy()[0].transpose(1, 2, 0))
                )
                plt.show()
