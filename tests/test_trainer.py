import csv
import os
import sys

import pytest
import torch
import wandb

sys.path.append("..")

from torchmate.trainer import Trainer  # noqa: E402


@pytest.fixture
def create_data():
    X_train = torch.rand(100, 1)
    y_train = torch.rand(100, 1)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10)

    X_val = torch.rand(20, 1)
    y_val = torch.rand(20, 1)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10)

    X_test = torch.rand(20, 1)
    test_dataset = torch.utils.data.TensorDataset(X_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10)
    return train_loader, val_loader, test_loader


@pytest.fixture
def create_model():
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = torch.nn.Linear(1, 1)

        def forward(self, x):
            return self.fc1(x)

    return SimpleModel


@pytest.fixture
def create_metrics():
    def mae(inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        mae = torch.abs(torch.mean(inputs - targets))
        return mae

    class MSE(torch.nn.Module):
        __name__ = "mse"

        def __init__(self, weight=None, size_average=True):
            super(MSE, self).__init__()

        def forward(self, inputs, targets):
            inputs = inputs.view(-1)
            targets = targets.view(-1)
            mse = torch.mean(torch.abs(inputs - targets))
            return mse

    return MSE, mae


@pytest.fixture
def create_trainer(create_data, create_model, create_metrics):
    train_loader, val_loader, _ = create_data
    SimpleModel = create_model
    MSE, mae = create_metrics

    model = SimpleModel()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    metrics = [MSE(), mae]

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        num_epochs=3,
        scheduler=scheduler,
        metrics=metrics,
    )
    return trainer


def test_trainer_fit(create_trainer):

    trainer = create_trainer

    history = trainer.fit()

    assert isinstance(history, dict)
    assert "loss" in history
    assert "val_loss" in history
    assert "mse" in history
    assert "val_mse" in history
    assert "mae" in history
    assert "val_mae" in history


def test_trainer_evaluate(create_data, create_model, create_metrics):

    train_loader, val_loader, _ = create_data
    SimpleModel = create_model
    MSE, mae = create_metrics

    model = SimpleModel()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, mode="min")
    metrics = [MSE(), mae]

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        num_epochs=3,
        scheduler=scheduler,
        metrics=metrics,
    )

    history = trainer.evaluate(metrics=[MSE()])

    assert isinstance(history, dict)
    assert "loss" not in history
    assert "val_loss" in history
    assert "mse" not in history
    assert "val_mse" in history
    assert "mae" not in history
    assert "val_mae" not in history


def test_trainer_predict(create_data, create_trainer):
    _, _, test_loader = create_data
    trainer = create_trainer
    predictons = trainer.predict(test_loader)
    assert len(predictons) == test_loader.batch_size * len(test_loader)
    assert isinstance(predictons, list)
    assert isinstance(predictons[0], torch.Tensor)
