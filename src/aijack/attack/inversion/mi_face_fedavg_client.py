"""Malicious client for FedAVG scheme"""

import copy

from .miface import MI_FACE

from ...manager import BaseManager
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
        model,
        user_id=0,
        lr=0.1,
        send_gradient=True,
        optimizer_type_for_global_grad="sgd",
        server_side_update=True,
        optimizer_kwargs_for_global_grad={},
        device="cpu",
        input_shape=(1, 1, 64, 64),
        target_label=0,
        lam=0.1,
        num_itr=100,
        beta=None,
        gamma=None,
        auxterm_func=lambda x: 0,
        process_func=lambda x: x,
        apply_softmax=False,
        log_interval=1,
        log_show_img=False,
        show_img_func=lambda x: x * 0.5 + 0.5,
        black_box=False
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

        self.miface = MI_FACE(
            self.model,
            input_shape=input_shape,
            target_label=target_label,
            lam=lam,
            num_itr=num_itr,
            beta=beta,
            gamma=gamma,
            auxterm_func=auxterm_func,
            process_func=process_func,
            apply_softmax=apply_softmax,
            device=device,
            log_interval=log_interval,
            log_show_img=log_show_img,
            show_img_func=show_img_func,
            black_box=black_box
        )

    def _setup_optimizer_for_global_grad(self, optimizer_type, **kwargs):
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

    def download(self, new_global_model):
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
        self, local_epoch, criterion, trainloader, optimizer, communication_id=0
    ):
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

        return loss_log


def attach_mpi_to_fedavgclient(cls):
    class MPIFedAVGClientWrapper(cls):
        def __init__(self, comm, *args, **kwargs):
            super(MPIFedAVGClientWrapper, self).__init__(*args, **kwargs)
            self.comm = comm

        def action(self):
            self.upload()
            self.model.zero_grad()
            self.download()

        def upload(self):
            self.upload_gradient()

        def upload_gradient(self, destination_id=0):
            self.comm.send(
                super(MPIFedAVGClientWrapper, self).upload_gradients(),
                dest=destination_id,
                tag=GRADIENTS_TAG,
            )

        def download(self):
            super(MPIFedAVGClientWrapper, self).download(
                self.comm.recv(tag=PARAMETERS_TAG)
            )

        def mpi_initialize(self):
            self.download()

    return MPIFedAVGClientWrapper


class MPIFedAVGClientManager(BaseManager):
    def attach(self, cls):
        return attach_mpi_to_fedavgclient(cls, *self.args, **self.kwargs)
