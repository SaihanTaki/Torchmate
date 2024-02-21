import csv
import os
import sys

import pytest
import torch
import wandb

sys.path.append("..")

from torchmate.callbacks import (  # noqa: E402
    CSVLogger,
    EarlyStopper,
    GradientAccumulator,
    GradientClipper,
    ModelCheckpoint,
    WandbLogger,
    WandbModelCheckpoint,
)


@pytest.fixture
def trainer():
    # Create a simple neural network model
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = torch.nn.Linear(1, 1)

        def forward(self, x):
            return self.fc1(x)

    # Create Mock Trainer Class
    class Trainer:
        def __init__(self):
            self.model = SimpleModel()
            self.train_dataloader = list(range(80))
            self.val_dataloader = list(range(20))
            self.loss_fn = torch.nn.MSELoss()
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
            self.history = dict().fromkeys(["loss", "val_loss", "acc", "val_acc"])
            self.early_stop = False
            self.accumulation_steps = 4
            self.update_params = True
            self.use_amp = False
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp and self.device != "cpu")

    return Trainer()


# CSVLogger


def test_csv_logger_initialization(tmpdir):
    filename = os.path.join(tmpdir, "test_logs/log.csv")
    csv_logger = CSVLogger(filename=filename)
    assert csv_logger.filename == filename
    assert csv_logger.separator == ","
    assert csv_logger.append is False
    assert csv_logger.current_epoch == 0


def test_csv_logger_on_experiment_begin(trainer, tmpdir):
    filename = os.path.join(tmpdir, "test_logs/log.csv")
    csv_logger = CSVLogger(filename=filename)
    csv_logger.on_experiment_begin(trainer)
    assert os.path.exists(os.path.join(tmpdir, "test_logs"))


def test_csv_logger_on_epoch_end(trainer, tmpdir):
    filename = os.path.join(tmpdir, "test_logs/log.csv")
    csv_logger = CSVLogger(filename=filename)
    csv_logger.on_experiment_begin(trainer)
    trainer.history["loss"] = [1.0, 0.5, 0.3]
    trainer.history["val_loss"] = [2.0, 0.7, 0.1]
    trainer.history["acc"] = [1.0, 0.5, 0.3]
    trainer.history["val_acc"] = [2.0, 0.7, 0.1]
    csv_logger.on_epoch_end(trainer)
    assert csv_logger.current_epoch == 1
    with open(filename, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["Epoch", "loss", "val_loss", "acc", "val_acc"]
        data = next(reader)
        assert data[0] == "1"
        assert float(data[1]) == 1.0


def test_csv_logger_on_experiment_end(trainer, tmpdir):
    filename = os.path.join(tmpdir, "test_logs/log.csv")
    csv_logger = CSVLogger(filename=filename)
    csv_logger.on_experiment_begin(trainer)
    csv_logger.on_experiment_end(trainer)
    assert os.path.exists(filename)


# EarlyStopper


def test_early_stopper_initialization():
    early_stopper = EarlyStopper()
    assert early_stopper.monitor == "val_loss"
    assert early_stopper.patience == 3
    assert early_stopper.mode == "min"
    assert early_stopper.min_delta == 0
    assert early_stopper.restore_best_state is False
    assert early_stopper.counter == 0
    assert early_stopper.epoch_count == 0
    assert early_stopper.optimum_value == float("inf")


def test_early_stopper_on_epoch_end_stop_min(trainer):
    early_stopper = EarlyStopper(monitor="val_loss", mode="min", patience=2)
    trainer.history["val_loss"] = [1.0, 1.1, 1.2]  # no improvement
    assert trainer.early_stop is False
    early_stopper.on_epoch_end(trainer)
    early_stopper.on_epoch_end(trainer)
    early_stopper.on_epoch_end(trainer)
    assert early_stopper.optimum_value == 1.0
    assert trainer.early_stop is True


def test_early_stopper_on_epoch_end_stop_max(trainer):
    early_stopper = EarlyStopper(monitor="val_acc", mode="max", patience=2)
    trainer.history["val_acc"] = [1.0, 0.9, 0.9]  # no improvement
    assert trainer.early_stop is False
    early_stopper.on_epoch_end(trainer)
    early_stopper.on_epoch_end(trainer)
    early_stopper.on_epoch_end(trainer)
    assert early_stopper.optimum_value == 1.0
    assert trainer.early_stop is True


def test_early_stopper_on_epoch_end_stop_min_delta(trainer):
    early_stopper = EarlyStopper(monitor="val_loss", mode="min", patience=2, min_delta=0.1)
    trainer.history["val_loss"] = [1.0, 0.9, 0.9]  # improvement but min_delta is 0.1
    assert trainer.early_stop is False
    early_stopper.on_epoch_end(trainer)
    early_stopper.on_epoch_end(trainer)
    early_stopper.on_epoch_end(trainer)
    assert early_stopper.optimum_value == 1.0
    assert trainer.early_stop is True


def test_early_stopper_on_epoch_end_continue(trainer):
    early_stopper = EarlyStopper(monitor="val_loss", mode="min", patience=2)
    trainer.history["val_loss"] = [1.0, 1.1, 0.9]  # improvement
    assert trainer.early_stop is False
    early_stopper.on_epoch_end(trainer)
    early_stopper.on_epoch_end(trainer)
    early_stopper.on_epoch_end(trainer)
    assert trainer.early_stop is False


def test_early_stopper_on_epoch_end_restore_best_state(trainer):
    early_stopper = EarlyStopper(monitor="val_loss", mode="min", patience=2)
    early_stopper.restore_best_state = True
    trainer.history["val_loss"] = [1.0, 1.1, 1.2]  # no improvement
    assert trainer.early_stop is False
    early_stopper.on_epoch_end(trainer)
    early_stopper.on_epoch_end(trainer)
    early_stopper.on_epoch_end(trainer)
    assert trainer.early_stop is True
    assert early_stopper.best_model_state is not None


# GradientAccumulator


def test_gradient_accumulator_initialization(trainer):
    gradient_accumulator = GradientAccumulator()
    assert gradient_accumulator.num_accum_steps is None
    assert gradient_accumulator.batch_count == 0
    assert trainer.update_params is True
    assert trainer.accumulation_steps == 4


def test_gradient_accumulator_on_experiment_begin(trainer):
    gradient_accumulator = GradientAccumulator(num_accum_steps=8)
    gradient_accumulator.on_experiment_begin(trainer)
    assert trainer.update_params is False
    assert trainer.accumulation_steps == 8


def test_gradient_accumulator_on_train_batch_begin(trainer):
    gradient_accumulator = GradientAccumulator(num_accum_steps=8)
    gradient_accumulator.on_experiment_begin(trainer)
    gradient_accumulator.on_train_batch_begin(trainer)  # 1st batch
    assert trainer.update_params is False
    for _ in range(6):
        gradient_accumulator.on_train_batch_begin(trainer)  # 2nd to 7th batch
    assert trainer.update_params is False
    gradient_accumulator.on_train_batch_begin(trainer)  # 8th batch
    assert trainer.update_params is True


# GradientClipper


def test_gradient_clipper_clip_by_value_initialization():
    gradient_clipper = GradientClipper(method="clip_by_value", clip_value=1.0)
    assert gradient_clipper.method == "clip_by_value"
    assert gradient_clipper.clip_value == 1.0
    assert gradient_clipper.max_norm is None


def test_gradient_clipper_clip_by_norm_initialization():
    gradient_clipper = GradientClipper(method="clip_by_norm", max_norm=1.0)
    assert gradient_clipper.method == "clip_by_norm"
    assert gradient_clipper.clip_value is None
    assert gradient_clipper.max_norm == 1.0


def test_gradient_clipper_invalid_method():
    with pytest.raises(ValueError):
        GradientClipper(method="invalid_method")


def test_gradient_clipper_missing_clip_value():
    with pytest.raises(ValueError):
        GradientClipper(method="clip_by_value")


def test_gradient_clipper_missing_max_norm():
    with pytest.raises(ValueError):
        GradientClipper(method="clip_by_norm")


def test_gradient_clipper_clip_by_value_on_backward_end(trainer):
    gradient_clipper = GradientClipper(method="clip_by_value", clip_value=1.0)
    X_train = torch.randn(1, 1)
    y_train = torch.randn(1, 1)
    with torch.autocast(
        device_type=trainer.device,
        dtype=torch.float16 if trainer.device != "cpu" else torch.bfloat16,
        enabled=trainer.use_amp and trainer.device != "cpu",
    ):
        y_pred = trainer.model(X_train)
        loss = trainer.loss_fn(y_pred, y_train)
        trainer.scaler.scale(loss).backward()
    gradient_clipper.on_backward_end(trainer)
    assert True  # If no exception, test passed


def test_gradient_clipper_clip_by_norm_on_backward_end(trainer):
    gradient_clipper = GradientClipper(method="clip_by_norm", max_norm=1.0)
    X_train = torch.randn(1, 1)
    y_train = torch.randn(1, 1)
    with torch.autocast(
        device_type=trainer.device,
        dtype=torch.float16 if trainer.device != "cpu" else torch.bfloat16,
        enabled=trainer.use_amp and trainer.device != "cpu",
    ):
        y_pred = trainer.model(X_train)
        loss = trainer.loss_fn(y_pred, y_train)
        trainer.scaler.scale(loss).backward()
    gradient_clipper.on_backward_end(trainer)
    assert True  # If no exception, test passed


# ModelCheckpoint


def test_model_checkpoint_initialization():
    checkpoint_dir = "test_checkpoints"
    checkpoint_callback = ModelCheckpoint(
        checkpoint_dir=checkpoint_dir,
    )
    assert checkpoint_callback.checkpoint_dir == checkpoint_dir
    assert checkpoint_callback.monitor == "val_loss"
    assert checkpoint_callback.mode == "min"
    assert checkpoint_callback.min_delta == 0.0
    assert checkpoint_callback.save_frequency == 1
    assert checkpoint_callback.save_best_only is True
    assert checkpoint_callback.save_state_dict_only is True
    assert checkpoint_callback.save_last is True
    assert checkpoint_callback.epoch_count == 0
    assert checkpoint_callback.optimum_value == float("inf")
    assert checkpoint_callback.best_checkpoint_path is None


def test_model_checkpoint_on_epoch_end_save_best_only_min(trainer, tmpdir):
    checkpoint_dir = str(tmpdir.join("checkpoints"))
    # os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        checkpoint_dir=checkpoint_dir,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )
    trainer.num_epochs = 3
    trainer.history["val_loss"] = [1.0, 0.5, 0.8]
    for epoch in range(1, trainer.num_epochs + 1):
        checkpoint_callback.on_epoch_end(trainer)
    best_checkpoint_path = checkpoint_callback.best_checkpoint_path
    assert os.path.exists(best_checkpoint_path)
    assert torch.load(best_checkpoint_path)["epoch"] == 2


def test_model_checkpoint_on_epoch_end_save_best_only_max(trainer, tmpdir):
    checkpoint_dir = str(tmpdir.join("checkpoints"))
    checkpoint_callback = ModelCheckpoint(
        checkpoint_dir=checkpoint_dir,
        monitor="val_acc",
        mode="max",
        save_best_only=True,
    )
    trainer.num_epochs = 3
    trainer.history["val_acc"] = [1.0, 0.5, 0.8]
    for epoch in range(1, trainer.num_epochs + 1):
        checkpoint_callback.on_epoch_end(trainer)
    best_checkpoint_path = checkpoint_callback.best_checkpoint_path
    assert os.path.exists(best_checkpoint_path)
    assert torch.load(best_checkpoint_path)["epoch"] == 1


def test_model_checkpoint_on_epoch_end_save_last(trainer, tmpdir):
    checkpoint_dir = str(tmpdir.join("checkpoints"))
    trainer.history["val_loss"] = [1.0, 0.5, 0.8]
    checkpoint_callback = ModelCheckpoint(
        checkpoint_dir=checkpoint_dir,
        save_last=True,
    )
    trainer.num_epochs = 3
    for epoch in range(1, trainer.num_epochs + 1):
        checkpoint_callback.on_epoch_end(trainer)
    last_checkpoint_path = os.path.join(checkpoint_dir, "ckpts", "model_last.pt")
    assert os.path.exists(last_checkpoint_path)
    assert torch.load(last_checkpoint_path)["epoch"] == 3


def test_model_checkpoint_on_epoch_end_save_frequency(trainer, tmpdir):
    checkpoint_dir = str(tmpdir.join("checkpoints"))
    trainer.history["val_loss"] = [1.0, 0.5, 0.8, 0.8, 0.7, 0.6, 0.4]
    checkpoint_callback = ModelCheckpoint(
        checkpoint_dir=checkpoint_dir,
        save_frequency=2,
        save_last=False,
        save_best_only=False,
    )
    trainer.num_epochs = 7
    for epoch in range(1, trainer.num_epochs + 1):
        checkpoint_callback.on_epoch_end(trainer)
    assert len(os.listdir(os.path.join(checkpoint_dir, "ckpts"))) == 3


# WandbLogger


def test_wandb_logger_on_experiment_begin(trainer):
    wandb_logger = WandbLogger()
    with pytest.raises(AssertionError):
        wandb_logger.on_experiment_begin(trainer)


def test_wandb_logger_on_epoch_end(trainer, mocker):
    mock_wandb_log = mocker.patch.object(wandb, "log")
    wandb_logger = WandbLogger()
    trainer.history = {"loss": [1.0, 0.8, 0.5]}
    wandb_logger.on_epoch_end(trainer)
    assert mock_wandb_log.called


# WandbModelCheckpoint


def test_wandb_model_checkpoint_initialization():
    checkpoint_dir = "checkpoints"
    wandb_checkpoint_callback = WandbModelCheckpoint(
        checkpoint_dir=checkpoint_dir,
    )
    assert wandb_checkpoint_callback.checkpoint_dir == checkpoint_dir
    assert wandb_checkpoint_callback.monitor == "val_loss"
    assert wandb_checkpoint_callback.mode == "min"
    assert wandb_checkpoint_callback.min_delta == 0.0
    assert wandb_checkpoint_callback.save_frequency == 1
    assert wandb_checkpoint_callback.save_best_only is True
    assert wandb_checkpoint_callback.save_state_dict_only is True
    assert wandb_checkpoint_callback.save_last is True
    assert wandb_checkpoint_callback.epoch_count == 0
    assert wandb_checkpoint_callback.optimum_value == float("inf")
    assert wandb_checkpoint_callback.best_checkpoint_path is None


def test_wandb_model_checkpoint_on_experiment_begin(trainer):
    wandb_checkpoint_callback = WandbModelCheckpoint(checkpoint_dir="checkpoints")
    with pytest.raises(AssertionError):
        wandb_checkpoint_callback.on_experiment_begin(trainer)


def test_wandb_model_checkpoint_on_epoch_end_save_best_only_min(trainer, tmpdir, mocker):
    checkpoint_dir = str(tmpdir.join("checkpoints"))
    mock_wandb_save = mocker.patch.object(wandb, "save")
    wandb_checkpoint_callback = WandbModelCheckpoint(
        checkpoint_dir=checkpoint_dir,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )
    trainer.num_epochs = 3
    trainer.history["val_loss"] = [1.0, 0.8, 0.5]
    for epoch in range(1, trainer.num_epochs + 1):
        wandb_checkpoint_callback.on_epoch_end(trainer)
    best_checkpoint_path = wandb_checkpoint_callback.best_checkpoint_path
    assert mock_wandb_save.called
    assert os.path.exists(best_checkpoint_path)
    assert torch.load(best_checkpoint_path)["epoch"] == 3


def test_wandb_model_checkpoint_on_epoch_end_save_best_only_max(trainer, tmpdir, mocker):
    checkpoint_dir = str(tmpdir.join("checkpoints"))
    mock_wandb_save = mocker.patch.object(wandb, "save")
    wandb_checkpoint_callback = WandbModelCheckpoint(
        checkpoint_dir=checkpoint_dir,
        monitor="val_acc",
        mode="max",
        save_best_only=True,
    )
    trainer.num_epochs = 3
    trainer.history["val_acc"] = [1.0, 0.8, 0.5]
    for epoch in range(1, trainer.num_epochs + 1):
        wandb_checkpoint_callback.on_epoch_end(trainer)
    best_checkpoint_path = wandb_checkpoint_callback.best_checkpoint_path
    assert mock_wandb_save.called
    assert os.path.exists(best_checkpoint_path)
    assert torch.load(best_checkpoint_path)["epoch"] == 1


def test_wandb_model_checkpoint_on_epoch_end_save_last(trainer, tmpdir, mocker):
    checkpoint_dir = str(tmpdir.join("checkpoints"))
    mock_wandb_save = mocker.patch.object(wandb, "save")
    trainer.history["val_loss"] = [1.0, 0.8, 0.5]
    wandb_checkpoint_callback = WandbModelCheckpoint(
        checkpoint_dir=checkpoint_dir,
        save_last=True,
    )
    trainer.num_epochs = 3
    for epoch in range(1, trainer.num_epochs + 1):
        wandb_checkpoint_callback.on_epoch_end(trainer)
    last_checkpoint_path = os.path.join(checkpoint_dir, "ckpts", "model_last.pt")
    assert mock_wandb_save.called
    assert os.path.exists(last_checkpoint_path)
    assert torch.load(last_checkpoint_path)["epoch"] == 3


def test_wandb_model_checkpoint_on_epoch_end_save_frequency(trainer, tmpdir, mocker):
    checkpoint_dir = str(tmpdir.join("checkpoints"))
    mock_wandb_save = mocker.patch.object(wandb, "save")
    trainer.history["val_loss"] = [1.0, 0.5, 0.8, 0.8, 0.7, 0.6, 0.4]
    wandb_checkpoint_callback = WandbModelCheckpoint(
        checkpoint_dir=checkpoint_dir,
        save_frequency=2,
        save_last=False,
        save_best_only=False,
    )
    trainer.num_epochs = 7
    for epoch in range(1, trainer.num_epochs + 1):
        wandb_checkpoint_callback.on_epoch_end(trainer)
    assert mock_wandb_save.called
    assert len(os.listdir(os.path.join(checkpoint_dir, "ckpts"))) == 3
