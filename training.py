from itertools import count
import numpy as np
import torch
import torch.nn
from sklearn.model_selection import train_test_split
from fast_data_loader import FastTensorDataLoader


def create_dataloaders(datasets, task, random_state, share_reduced_training=1.0):
    datatype = torch.float32
    X = datasets[task]['X']
    y = datasets[task]['y']
    train_val_indices, test_indices = train_test_split(
        np.arange(len(X)),
        test_size=0.15,
        random_state=random_state,
        shuffle=True
    )

    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=0.15,
        random_state=random_state,
        shuffle=True
    )

    train_indices_reduced = train_indices[:int(len(train_indices) * share_reduced_training)]
    train_dataloader = FastTensorDataLoader(
        torch.tensor(X[train_indices_reduced], dtype=datatype),
        torch.tensor(y[train_indices_reduced], dtype=datatype)
    )

    val_dataloader = FastTensorDataLoader(
        torch.tensor(X[val_indices], dtype=datatype),
        torch.tensor(y[val_indices], dtype=datatype)
    )

    test_dataloader = FastTensorDataLoader(
        torch.tensor(X[test_indices], dtype=datatype),
        torch.tensor(y[test_indices], dtype=datatype)
    )

    return train_dataloader, val_dataloader, test_dataloader, train_indices_reduced, val_indices, test_indices


def calculate_validation_loss(model: torch.nn.Module, dataloader, task):
    criterion = torch.nn.MSELoss(reduction='mean')
    mode = model.training
    model.eval()
    with torch.no_grad():
        x, y = dataloader.tensors
        out = model(x, head=task)
        loss = criterion(out, y).item()
    model.train(mode=mode)
    return loss


def training_loop(model: torch.nn.Module, lr: float, train_dataloader, val_dataloader, test_dataloader, task, lamb,
                  patience=20,
                  epochs=10000, use_omega=False):
    min_val_loss = np.Inf
    initial_val_loss = np.Inf
    epochs_since_min = 0
    min_val_loss_state = model.state_dict()

    train_losses = []
    train_losses_with_omega = []
    val_losses = []

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    mode = model.training

    for e in count(start=0):
        # skip training for first iteration (validation only)
        if e > 0:

            model.train()

            for x, y in train_dataloader:
                optimizer.zero_grad()
                out = model(x, head=task)
                loss = criterion(out, y)
                omega_loss = lamb * model.compute_omega_loss()

                train_losses.append(loss.item())
                train_losses_with_omega.append(omega_loss.item())

                if use_omega:
                    overall_loss = loss + omega_loss
                else:
                    overall_loss = loss

                overall_loss.backward()
                optimizer.step()

        with torch.no_grad():
            model.eval()

            x, y = val_dataloader.tensors

            out = model(x, head=task)

            loss = criterion(out, y)
            val_loss = loss.item()
            initial_val_loss = val_loss

            val_losses.append(val_loss)

        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            epochs_since_min = 0
            min_val_loss_state = model.state_dict()
        else:
            epochs_since_min += 1
        model.train(mode=mode)

        # check for early stopping or whether the number of specified epochs is reached
        if epochs_since_min == patience or e == epochs:
            if e == epochs:
                print("Maximum number of epochs reached")
            model.load_state_dict(min_val_loss_state)

            with torch.no_grad():
                model.eval()

                x, y = test_dataloader.tensors

                out = model(x, head=task)

                loss = criterion(out, y)
                test_loss = loss.item()
                model.train(mode=mode)
            return test_loss, min_val_loss, initial_val_loss, train_losses, train_losses_with_omega, val_losses
