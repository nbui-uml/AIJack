"""Malicious client for FedAVG scheme"""

from __future__ import annotations

import copy
import pickle

import numpy as np
import torch
import torch.types
import torch.utils
import torch.utils.data

from .miface import MI_FACE

from ...collaborative.core import BaseClient
from ...collaborative.core.utils import GRADIENTS_TAG, PARAMETERS_TAG
from ...collaborative.optimizer import AdamFLOptimizer, SGDFLOptimizer


class MIFaceFedAVGClient(BaseClient):
    """Client of FedAVG for single process simulation

    Args:
        model (torch.nn.Module): local model
        user_id (int, optional): if of this client. Defaults to 0.
        lr (float, optional): learning rate. Defaults to 0.1.
        send_gradient (bool, optional): if True, communicate gradient to the server. otherwise, communicates model parameters. Defaults to True.
        optimizer_type_for_global_grad (str, optional): type of optimizer for model update with global gradient. sgd|adam. Defaults to "sgd".
        server_side_update (bool, optional): If True, the global model update is conducted in the server side. Defaults to True.
        optimizer_kwargs_for_global_grad (dict, optional): kwargs for the optimizer for global gradients. Defaults to {}.
        device (str, optional): device type. Defaults to "cpu".
    """

    def __init__(
        self,
        model: torch.nn.Module,
        user_id=0,
        lr=0.1,
        send_gradient=True,
        optimizer_type_for_global_grad="sgd",
        server_side_update=True,
        optimizer_kwargs_for_global_grad={},
        device="cpu",
    ):
        super(MIFaceFedAVGClient, self).__init__(model, user_id=user_id)
        self.lr = lr
        self.send_gradient = send_gradient
        self.server_side_update = server_side_update
        self.device = device

        if not self.server_side_update:
            self._setup_optimizer_for_global_grad(
                optimizer_type_for_global_grad, **optimizer_kwargs_for_global_grad
            )

        self.prev_parameters = []
        for param in self.model.parameters():
            self.prev_parameters.append(copy.deepcopy(param))

        self.initialized = False

        self.epoch = 0
        self.mi_face = None
        self.mi_logfn = None
        self.mi_start_epoch = 1
        self.mi_atk_interval = 1
        self.mi_num_atk = 0

    def _setup_optimizer_for_global_grad(self, optimizer_type: str, **kwargs):
        if optimizer_type == "sgd":
            self.optimizer_for_gloal_grad = SGDFLOptimizer(
                self.model.parameters(), lr=self.lr, **kwargs
            )
        elif optimizer_type == "adam":
            self.optimizer_for_gloal_grad = AdamFLOptimizer(
                self.model.parameters(), lr=self.lr, **kwargs
            )
        elif optimizer_type == "none":
            self.optimizer_for_gloal_grad = None
        else:
            raise NotImplementedError(
                f"{optimizer_type} is not supported. You can specify `sgd`, `adam`, or `none`."
            )

    def attach_mi_face(
        self, mi_face: MI_FACE, log_fn: str, start_epoch=1, atk_interval=1, num_atk=1
    ):
        """
        Attach MI_Face attack to client

        Args:
            mi_face (MI_Face): MI_Face API object
            log_fn (str | path): file to store pickled MIFaceFedAVGLog object
            start_epoch (int): epoch to start launching attack
            atk_interval (int): number of epochs between consecutive attacks
            num_attack (int): number of attacks to perform total
        """
        self.mi_face = mi_face
        self.mi_logfn = log_fn
        self.mi_start_epoch = start_epoch
        self.mi_atk_interval = atk_interval
        self.mi_num_atk = num_atk
        self.mi_log = MIFaceFedAVGLog()

    def upload(self):
        """Upload the current local model state"""
        if self.send_gradient:
            return self.upload_gradients()
        else:
            return self.upload_parameters()

    def upload_parameters(self):
        """Upload the model parameters"""
        return self.model.state_dict()

    def upload_gradients(self):
        """Upload the local gradients"""
        gradients = []
        for param, prev_param in zip(self.model.parameters(), self.prev_parameters):
            gradients.append((prev_param - param) / self.lr)
        return gradients

    def revert(self):
        """Revert the local model state to the previous global model"""
        for param, prev_param in zip(self.model.parameters(), self.prev_parameters):
            if param is not None:
                param = prev_param

                # decrement the epoch count to the previous model's
                self.epoch += -1

    def download(self, new_global_model: torch.nn.Module):
        """Download the new global model"""
        if self.server_side_update or (not self.initialized):
            # receive the new global model as the model state
            self.model.load_state_dict(new_global_model)
        else:
            # receive the new global model as the global gradients
            self.revert()
            self.optimizer_for_gloal_grad.step(new_global_model)

        if not self.initialized:
            self.initialized = True

        self.prev_parameters = []
        for param in self.model.parameters():
            self.prev_parameters.append(copy.deepcopy(param))

    def local_train(
        self,
        local_epoch: int,
        criterion: function,
        trainloader: torch.utils.data.DataLoader,
        optimizer,
        communication_id=0,
    ):
        """
        Train local model on data and perform attack if applicable.
        """
        loss_log = []

        for _ in range(local_epoch):
            running_loss = 0.0
            running_data_num = 0
            for _, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                self.zero_grad()

                outputs = self(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_data_num += inputs.shape[0]

            loss_log.append(running_loss / running_data_num)

        self.epoch += 1
        if (
            self.mi_num_atk > 0
            and self.epoch >= self.mi_start_epoch
            and (self.epoch - self.mi_start_epoch) % self.mi_atk_interval == 0
        ):
            im, log = self.mi_face.attack()
            self.mi_log.append(MIFaceFedAVGEntry(im, min(log), self.epoch))
            with open(self.mi_logfn, "wb") as fout:
                pickle.dump(self.mi_log, fout)

        return loss_log


class MIFaceFedAVGLog(list):
    def __init__(self):
        super(MIFaceFedAVGLog, self).__init__()

    def append(self, entry: MIFaceFedAVGEntry):
        super(MIFaceFedAVGLog, self).append(entry)


class MIFaceFedAVGEntry:
    def __init__(self, im: np.typing.ArrayLike, c: float, epoch: int):
        self.im = im
        self.c = c
        self.epoch = epoch
