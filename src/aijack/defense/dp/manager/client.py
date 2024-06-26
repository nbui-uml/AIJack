import torch
from torch.utils.data import TensorDataset

from ....manager import BaseManager


def attach_dpsgd_to_client(cls, privacy_manager, sigma):
    """
    Attaches DPSGD (Differentially Private Stochastic Gradient Descent) functionality to the client class.

    Args:
        cls: Client class to which DPSGD functionality will be attached.
        privacy_manager: Privacy manager object providing DPSGD functionality.
        sigma (float): Noise multiplier for privacy.

    Returns:
        tuple: Tuple containing the DPSGDClientWrapper class and the privacy optimizer wrapper.
    """
    dpoptimizer_wrapper, lot_loader, batch_loader = privacy_manager.privatize(sigma)

    class DPSGDClientWrapper(cls):
        def __init__(self, *args, **kwargs):
            super(DPSGDClientWrapper, self).__init__(*args, **kwargs)
            self.privacy_manager = privacy_manager

        def local_train(
            self, local_epoch, criterion, trainloader, optimizer, communication_id=0
        ):
            _ = trainloader  # we do not explicitly use it

            for i in range(local_epoch):
                running_loss = 0.0
                running_data_num = 0
                for X_lot, y_lot in lot_loader(optimizer):
                    for X_batch, y_batch in batch_loader(TensorDataset(X_lot, y_lot)):
                        optimizer.zero_grad()
                        pred = self(X_batch)
                        loss = criterion(pred, y_batch.to(torch.int64))
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                        running_data_num += X_batch.shape[0]

                print(
                    f"communication {communication_id}, epoch {i}: client-{self.user_id+1}",
                    running_loss / running_data_num,
                )

    return DPSGDClientWrapper, dpoptimizer_wrapper


class DPSGDClientManager(BaseManager):
    """
    Manager class for attaching DPSGD to clients.
    """

    def attach(self, cls):
        """
        Attaches DPSGD to the client class.

        Args:
            cls: Client class.

        Returns:
            DPSGDClientWrapper: Wrapped client class with DPSGD functionality.
        """
        return attach_dpsgd_to_client(cls, *self.args, **self.kwargs)
