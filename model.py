import copy

import numpy as np
import torch
import torch.nn as nn

from training import calculate_validation_loss


class MAS(nn.Module):
    """
    Multi-task multilayer perceptron model.

    Multilayer perceptron model with support for multiple output heads (tasks).
    Arbitrary number of hidden layers with individual numbers of neurons for each layer.
    The output dimension is the same across all output heads.

    Attributes:
        layers: Dictionary containing all network layers.
        layer_dims: List of tuples of input and output dimensions of hidden and output layers.
        n_hidden: integer describing number of hidden layers.
        activation: PyTorch activation function.
    """

    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, heads: []):
        """
        Generates and initializes the network.

        Args:
            input_dim: Integer value describing the number of input features (dimensionality of the input).
            hidden_dims: List of integer values describing the number of neurons for each corresponding hidden layer.
            output_dim: Integer value describing the number of output features (dimensionality of the output)
        """

        super(MAS, self).__init__()

        self.n_hidden = len(hidden_dims)
        self.heads = heads

        # create pairs of input/output dimensions for each layer
        self.layer_dims = list(zip([input_dim] + hidden_dims,  # input dimensions
                                   hidden_dims + [output_dim]))  # output dimension

        self.layers = nn.ModuleDict()

        # create hidden layers
        for i, dims in enumerate(self.layer_dims[:-1], 1):
            self.layers.add_module('h{}'.format(i), nn.Linear(*dims))

        for head in heads:
            self.layers.add_module(f'out_{head}', nn.Linear(in_features=hidden_dims[-1], out_features=1))

        self.init_omega_and_theta()

        self.activation = nn.LeakyReLU()

    def init_omega_and_theta(self):
        """Initializes omega and theta buffers for the corresponding network parameters."""

        # known omega and theta values (in case of previously initialized buffers)
        omega_dict = {name: buff for name, buff in self.named_buffers() if name.startswith('omega')}
        theta_dict = {name: buff for name, buff in self.named_buffers() if name.startswith('theta')}

        # initialize weight importance omega and reference weights theta associated with omega
        for name, param in self.named_parameters():
            # omega values are initialized with zeros
            if f"omega_{name.replace('.', '-')}" not in omega_dict:
                self.register_buffer(f"omega_{name.replace('.', '-')}", torch.zeros_like(param, requires_grad=False))
            # theta values are initialized using the current weight values
            if f"theta_{name.replace('.', '-')}" not in theta_dict:
                self.register_buffer(f"theta_{name.replace('.', '-')}", param.clone().detach())

    def forward(self, x, head: str):
        """Forward pass through the network.

        Args:
            x: Input to the network (must match dimension specified by the attribute 'input_dim')
            head: Output head used in forward pass.

        Returns:
            Output of the network
        """
        for i in range(self.n_hidden):
            x = self.layers['h{}'.format(i + 1)](x)
            x = self.activation(x)
        x = self.layers[f'out_{head}'](x)
        return x

    def add_head(self, head, dataloader, clone=True):
        min_loss = np.inf
        task_with_min_loss = None
        statistics = {'losses': {}, 'head': head}
        if clone:
            # calculate best existing head and use cloned weights for new head
            for existing_head in self.heads:
                validation_loss = calculate_validation_loss(model=self, dataloader=dataloader, task=existing_head)
                statistics['losses'][existing_head] = validation_loss
                if validation_loss < min_loss:
                    min_loss = validation_loss
                    task_with_min_loss = existing_head

            if task_with_min_loss:
                statistics['used'] = task_with_min_loss
                # print(f'cloned layer: {task_with_min_loss}')
                self.heads.append(head)

                self.layers.add_module(f'out_{head}', copy.deepcopy(self.layers[f'out_{task_with_min_loss}']))
                self.init_omega_and_theta()
                return statistics

        # create new head without cloning
        statistics['used'] = 'new'
        self.heads.append(head)

        dims = self.layer_dims[-1]
        self.layers.add_module(f'out_{head}', nn.Linear(*dims))
        self.init_omega_and_theta()
        return statistics


    def add_head_finetuning(self, head, dataloader): # Clone last existing head (without searching best head)
        min_loss = np.inf
        task_with_min_loss = None
        statistics = {'losses': {}, 'head': head}
        if self.heads:
            last_task = self.heads[-1]
            statistics['used'] = last_task
            self.heads.append(head)
            self.layers.add_module(f'out_{head}', copy.deepcopy(self.layers[f'out_{last_task}']))
            self.init_omega_and_theta()
            return statistics

        # create new head
        statistics['used'] = 'new'
        self.heads.append(head)

        dims = self.layer_dims[-1]
        self.layers.add_module(f'out_{head}', nn.Linear(*dims))
        self.init_omega_and_theta()
        return statistics

    def add_head_and_get_cloned_task(self, head, dataloader, clone=True):
        min_loss = np.inf
        task_with_min_loss = None
        statistics = {'losses': {}, 'head': head}
        if clone:
            # calculate best existing head and use cloned weights for new head
            for existing_head in self.heads:
                validation_loss = calculate_validation_loss(model=self, dataloader=dataloader, task=existing_head)
                statistics['losses'][existing_head] = validation_loss
                if validation_loss < min_loss:
                    min_loss = validation_loss
                    task_with_min_loss = existing_head

            if task_with_min_loss:
                statistics['used'] = task_with_min_loss
                # print(f'cloned layer: {task_with_min_loss}')
                self.heads.append(head)

                self.layers.add_module(f'out_{head}', copy.deepcopy(self.layers[f'out_{task_with_min_loss}']))
                self.init_omega_and_theta()
                return statistics, task_with_min_loss

        # create new head without cloning
        statistics['used'] = 'new'
        self.heads.append(head)

        dims = self.layer_dims[-1]
        self.layers.add_module(f'out_{head}', nn.Linear(*dims))
        self.init_omega_and_theta()
        return statistics, "new"

    def update_theta(self):
        """Updates theta buffers using the current weight values."""

        # get theta buffers
        theta_dict = {name: buff for name, buff in self.named_buffers() if name.startswith('theta')}

        for name, param in self.named_parameters():
            # get matching theta value
            theta = theta_dict['theta_{}'.format(name.replace('.', '-'))]

            # clone current parameter values
            theta.data = param.clone().detach()

    def compute_omega_loss(self):
        """
        Computes the MAS loss based on the omega and theta buffers.

        Returns:
            Float value describing the MAS loss.
        """
        omega_loss = 0.0

        for name, param in self.named_parameters():
            # get matching omega and theta values
            omega = self._buffers['omega_{}'.format(name.replace('.', '-'))]
            theta = self._buffers['theta_{}'.format(name.replace('.', '-'))]

            # sum up squared differences in the parameters
            omega_loss += torch.sum(((param - theta) ** 2)
                                    * omega
                                    )

        return omega_loss

    def update_omega(self, dataloader, task, gamma, reset_omegas=False):
        criterion = torch.nn.MSELoss(reduction='mean')
        self.zero_grad()

        mode = self.training
        self.eval()

        if reset_omegas:
            # initialize weight importance omega and reference weights theta associated with omega
            for name, param in self.named_parameters():
                # omega values are initialized with zeros
                self.register_buffer(f"omega_{name.replace('.', '-')}", torch.zeros_like(param, requires_grad=False))

        self.zero_grad()
        x, y = dataloader.tensors

        out = self(x, head=task)
        loss = criterion(out, torch.zeros(out.size()))
        loss.backward()

        omega_dict = {name: buff for name, buff in self.named_buffers() if name.startswith('omega')}

        # average gradients over number of samples and add to omega
        for name, param in self.named_parameters():

            # get matching omega value
            omega = omega_dict['omega_{}'.format(name.replace('.', '-'))]

            # check if gradient is available (not the case for unused output heads)
            if param.grad is not None:
                # decay previous omega values using gamma
                omega.data *= gamma
                # add new omega values
                omega.data += torch.abs(param.grad.detach())

        self.train(mode=mode)
