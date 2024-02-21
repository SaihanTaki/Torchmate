import re
from typing import Callable, List, Optional, Type, Union

import matplotlib.pyplot as plt
import torch

import torchmate
from torchmate.callbacks import Callback
from torchmate.utils import ProgressBar, RunningAverage, colorize_text


class Trainer(torch.nn.Module):
    """Encapsulate training essentials

    Args:
        model (torch.nn.Module, required): The PyTorch model to be trained.
        train_dataloader (torch.utils.data.DataLoader, required): DataLoader for the training dataset.
        val_dataloader (torch.utils.data.DataLoader, required): DataLoader for the validation dataset.
        loss_fn (torch.nn.Module, required): Loss function for training.
        optimizer (torch.optim.Optimizer, required): Optimizer for updating model parameters.
        num_epochs (int, optional): Number of training epochs (default is 1).
        test_dataloader (torch.utils.data.DataLoader, optional): DataLoader for the test dataset.
        metrics (List[callable], optional): List of metrics functions for evaluation.
        callbacks (List[Callback], optional): List of callback functions for various stages.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
        schedule_monitor (str, optional): Metric to monitor for scheduler (default is "val_loss").
        mixed_precision (bool, optional): Whether to use mixed precision (fp16) training (default is False).
        use_gradient_penalty (bool, optional): Whether to use gradient penalty (default is False).
        device (str, optional): Device to use for training (default is "cpu").

    Other Attributes:
        - **history (dict):** Training history containing loss, metrics, and learning rates.
        - **early_stop (bool):** Flag for early stopping.
        - **update_params (bool):** Flag for updating model parameters.
        - **accumulation_steps (int):** Number of steps for gradient accumulation during training.

    Important Methods:
        - **fit():** Train and validate the model for the specified number of epochs and return history.
        - **evaluate():** Evaluate the model on the validation dataset and return evaluation history.
        - **predict():** Make predictions using the model on the test dataset.

    **Example usage:**

    .. code-block:: python

        import torch
        import numpy as np

        import os
        import time

        from torchmate.trainer import Trainer
        from torchmate.callbacks import CSVLogger, ModelCheckpoint
        from sklearn.model_selection import train_test_split

        # Create a simple neural network model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc1 = torch.nn.Linear(1, 1)

            def forward(self, x):
                return self.fc1(x)

        # Create synthetic data
        X = torch.tensor(np.random.rand(1000, 1), dtype=torch.float32)
        y = 2 * X + 1 + torch.randn(1000, 1) * 0.1  # Adding some noise

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create DataLoader objects for training and validation
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)


        # Create Metrics

        class MSE(torch.nn.Module):
            __name__ = 'mse'
            def __init__(self, weight=None, size_average=True):
                super(MSE, self).__init__()
            def forward(self, inputs, targets):
                inputs = inputs.view(-1)
                targets = targets.view(-1)
                mse = torch.mean(torch.abs(inputs - targets))
                return mse


        def mae(inputs, targets):
            inputs = inputs.view(-1)
            targets = targets.view(-1)
            mae = torch.abs(torch.mean(inputs - targets))
            return mae

        model = SimpleModel()
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

        metrics = [MSE(),mae]

        logdir = "logs"
        csv_file = os.path.join(logdir,"logs.csv")
        ckpt_dir = os.path.join(logdir,"model")

        callbacks = [CSVLogger(filename=csv_file),
                    ModelCheckpoint(checkpoint_dir=ckpt_dir)
                    ]


        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(device)

        # Create a Trainer instance with the callbacks
        trainer = Trainer(model,
                        train_dataloader,
                        val_dataloader,
                        loss_fn,
                        optimizer,
                        num_epochs=3,
                        scheduler=scheduler,
                        metrics=metrics,
                        callbacks=callbacks,
                        device=device,
                        mixed_precision=True,
                        use_grad_penalty=True
                        )


        # Train the model
        history = trainer.fit()

        print("_"*150)

        print(pd.read_csv(csv_file))

    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        loss_fn: Union[Callable, torch.nn.Module],
        optimizer: torch.optim.Optimizer,
        num_epochs: int = 1,
        test_dataloader: Optional[torch.utils.data.DataLoader] = None,
        metrics: Optional[List[Callable]] = None,
        callbacks: Optional[List[Type[torchmate.callbacks.Callback]]] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        schedule_monitor: str = "val_loss",
        mixed_precision: bool = False,
        use_grad_penalty: bool = False,
        device: Union[str, torch.device] = "cpu",
    ):

        super().__init__()
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.schedule_monitor = schedule_monitor
        self.metrics = metrics
        self.num_epochs = num_epochs
        self.callbacks = callbacks
        self.use_amp = mixed_precision
        self.use_grad_penalty = use_grad_penalty
        self.device = device
        self.history = {}
        self.early_stop = False
        self.update_params = True
        self.accumulation_steps = 4
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp and self.device != "cpu")

        # Initializig the log history dict
        self.history["loss"] = []
        self.history["val_loss"] = []
        self.history["lr"] = []
        if self.metrics is not None:
            for metric in self.metrics:
                self.history[f"{metric.__name__}"] = []
                self.history[f"val_{metric.__name__}"] = []
        ########################################################

        # set __name__ attribute for metrics and loss. It is import for logging and printing
        if self.metrics is not None:
            for metric in self.metrics:
                if not hasattr(metric, "__name__"):
                    metric.__name__ = self.camel_to_snake_case(metric.__class__.__name__)

        if not hasattr(self.loss_fn, "__name__"):
            self.loss_fn.__name__ = self.camel_to_snake_case(self.loss_fn.__class__.__name__)
        ##########################################################################################

    def fit(self):
        """Train the model and returns the training history.

        Returns:
            Dict : A dictionary object encapsulating the training history.
        """
        history = self.train_and_evaluate()
        return history

    def evaluate(
        self,
        dataloader: Optional[torch.utils.data.DataLoader] = None,
        loss_fn: Union[Callable, torch.nn.Module] = None,
        metrics: Optional[List[Callable]] = None,
        callbacks: Optional[List[Type[torchmate.callbacks.Callback]]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Evaluate the model on the a dataset and returns the evaluation history.

        This method provides flexibility for customized evaluation.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): A PyTorch DataLoader containing the validation data.
                If not provided, the ``self.val_dataloader`` attribute will be used. Defaults to ``None``.
            loss_fn (Callable or torch.nn.Module, optional): A custom loss function for evaluation.
                If not provided, the ``self.loss_fn`` attribute will be used. Defaults to ``None``.
            metrics (List[Callable], optional): A list of custom evaluation metrics.
                If not provided, the ``self.metrics`` attribute will be used. Defaults to ``None``.
            callbacks (List[Callback], optional): A list of callback objects for evaluation stages.
                If not provided, the ``self.callbacks`` attribute will be used. Defaults to ``None``.
            device (str or torch.device, optional): The device to use for evaluation (e.g., "cpu" or "cuda").
                If not provided, the ``self.device`` attribute will be used. Defaults to ``None``.

        Returns:
            Dict: A dictionary object containing the evaluation results.
        """

        model = self.model
        dataloader = dataloader if dataloader else self.val_dataloader
        callbacks = callbacks if callbacks else self.callbacks
        loss_fn = loss_fn if loss_fn else self.loss_fn
        metrics = metrics if metrics else self.metrics
        device = device if device else self.device

        history = self.evaluate_single_epoch(model, dataloader, loss_fn, metrics, callbacks, device)
        return history

    def predict(
        self,
        test_dataloader: Optional[torch.utils.data.DataLoader] = None,
        callbacks: Optional[List[Type[torchmate.callbacks.Callback]]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Perform predictions on the provided test data using the trained model.

        This method enables you to make predictions on a test dataset using the trained model within your `Trainer` class.

        Args:
            test_dataloader (DataLoader, optional): A PyTorch DataLoader containing the test data.
                If not provided, the ``self.test_dataloader`` attribute will be used. Defaults to ``None``.
            callbacks (list[Callback], optional): A list of callback objects to be executed at various stages
                of the prediction process. If not provided, the ``self.callbacks`` attribute will be used.
                Defaults to ``None``.
            device (str or torch.device, optional): The device to run the prediction on (e.g., "cpu" or "cuda").
                If not provided, the ``self.device`` attribute will be used. Defaults to ``None``.

        Returns:
            torch.Tensor: A PyTorch Tensor containing the predicted outputs for the test data.

        Raises:
            ValueError: If both ``test_dataloader`` and ``self.test_dataloader`` are None.

        """

        if test_dataloader is None and self.test_dataloader is None:
            raise ValueError(
                "Missing validation data: You must provide either a `test_dataloader` argument or set a \
                `test_dataloader` attribute on the Trainer instance."
            )

        model = self.model
        device = device if device else self.device
        callbacks = callbacks if callbacks else self.callbacks
        device = device if device else self.device

        if test_dataloader:
            self.test_dataloader = test_dataloader
        else:
            test_dataloader = self.test_dataloader

        model.eval()
        progress_bar = ProgressBar(total=len(test_dataloader), prefix="prediction")
        self.execute_callbacks(self, self.callbacks, "predict_begin")
        predictions = []
        for batch_ix, X_test in enumerate(test_dataloader):
            self.execute_callbacks(self, self.callbacks, "predict_batch_begin")
            X_test = X_test[0].to(device)
            with torch.inference_mode():
                y_pred = self.model(X_test)
                predictions += y_pred
            progress_bar.update(batch_ix + 1)
            self.execute_callbacks(self, self.callbacks, "predict_batch_end")
        self.execute_callbacks(self, self.callbacks, "predict_end")
        return predictions

    @staticmethod
    def camel_to_snake_case(text: str) -> str:
        """Convert CamelCase text to snake_case."""
        string = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", text)
        string = re.sub("([a-z0-9])([A-Z])", r"\1_\2", string)
        return string.lower()

    @staticmethod
    def gradient_norm(model, loss):
        # Creates gradients
        grad_params = torch.autograd.grad(outputs=loss, inputs=model.parameters(), create_graph=True)
        # Computes the penalty term and adds it to the loss
        grad_norm = 0
        for grad in grad_params:
            grad_norm += grad.pow(2).sum()
        grad_norm = grad_norm.sqrt()
        return grad_norm

    @staticmethod
    def execute_callbacks(trainer, callbacks=None, stage=""):

        if callbacks is None:
            return None

        valid_stages = [
            "experiment_begin",
            "experiment_end",
            "epoch_begin",
            "epoch_end",
            "train_begin",
            "train_end",
            "train_batch_begin",
            "train_batch_end",
            "val_begin",
            "val_end",
            "val_batch_begin",
            "val_batch_end",
            "predict_begin",
            "predict_end",
            "predict_batch_begin",
            "predict_batch_end",
            "backward_end",
        ]

        if stage not in valid_stages:
            raise ValueError(f"Invalid stage name. Must be one of {valid_stages}")

        for callback in callbacks:
            method = f"on_{stage}"
            if hasattr(callback, method):
                callback_method = getattr(callback, method)
                callback_method(trainer)
        return None

    def train_and_evaluate(self):

        self.model.to(self.device)
        # running_avg_dict = dict() // assigned but never used, delete it
        History = dict()
        History["loss"] = []
        History["val_loss"] = []
        History["lr"] = []

        if self.metrics is not None:
            for metric in self.metrics:
                History[f"{metric.__name__}"] = []
                History[f"val_{metric.__name__}"] = []

        self.execute_callbacks(self, self.callbacks, "experiment_begin")
        for epoch in range(1, self.num_epochs + 1):
            etxt = f"Epoch {epoch}/{self.num_epochs}"
            etxt = colorize_text(etxt, fore_tuple=(0, 0, 255), bold_text=True)
            print(etxt, end="\n")
            self.execute_callbacks(self, self.callbacks, "epoch_begin")

            history = self.train_single_epoch(
                self.model,
                self.train_dataloader,
                self.optimizer,
                self.loss_fn,
                self.metrics,
                self.callbacks,
                self.device,
            )

            val_history = self.evaluate_single_epoch(
                self.model, self.val_dataloader, self.loss_fn, self.metrics, self.callbacks, self.device
            )

            # update history
            for key in history.keys():
                History[key].append(history[key])
            for key in val_history.keys():
                History[key].append(val_history[key])
            self.history = History
            #########################################################

            self.execute_callbacks(self, self.callbacks, "epoch_end")
            if self.early_stop:
                break
            if self.update_params:  # schedule learning rate only when the parameters are updated
                if self.scheduler is not None:
                    if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
                        self.scheduler.step(val_history[self.schedule_monitor])
                    else:
                        self.scheduler.step()
        self.execute_callbacks(self, self.callbacks, "experiment_end")

        return History

    def train_single_epoch(
        self, model, train_dataloader, optimizer, loss_fn, metrics=None, callbacks=None, device=None
    ):

        model.train()

        progress_bar = ProgressBar(total=len(train_dataloader), prefix="train")
        history = dict()
        loss_avg = RunningAverage()
        running_avg_dict = dict()

        if metrics is not None:
            for metric in metrics:
                running_avg_dict[f"{metric.__name__}_avg"] = RunningAverage()

        self.execute_callbacks(self, callbacks, "train_begin")
        for batch_ix, (X_train, y_train) in enumerate(train_dataloader):
            self.execute_callbacks(self, callbacks, "train_batch_begin")
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            with torch.autocast(
                device_type=device,
                dtype=torch.float16 if self.device != "cpu" else torch.bfloat16,
                enabled=self.use_amp and self.device != "cpu",
            ):
                y_pred = model(X_train)
                batch_loss = loss_fn(y_pred, y_train)
                if self.accumulation_steps > 1 and not self.update_params:
                    batch_loss = batch_loss / self.accumulation_steps
                if self.use_grad_penalty:
                    batch_loss = batch_loss + self.gradient_norm(model, batch_loss)

            self.scaler.scale(batch_loss).backward()  # batch_loss.backward()
            if self.update_params:
                self.execute_callbacks(self, callbacks, "backward_end")
                self.scaler.step(optimizer)
                self.scaler.update()  # optimizer.step()
                optimizer.zero_grad()
            # update value + message
            loss_avg.update(batch_loss.item())
            message = f"loss: {round(loss_avg(),5)}"
            if metrics is not None:
                for metric in metrics:
                    running_avg_dict[f"{metric.__name__}_avg"].update(metric(y_pred, y_train).item())
                    metric_value = round(running_avg_dict[f"{metric.__name__}_avg"](), 5)
                    message += f" | {metric.__name__}: {metric_value}"
            ###############################################
            self.execute_callbacks(self, callbacks, "train_batch_end")
            progress_bar.update(batch_ix + 1, message)
        self.execute_callbacks(self, callbacks, "train_end")

        # update history
        history["lr"] = self.optimizer.param_groups[0]["lr"]
        history["loss"] = loss_avg()
        if metrics is not None:
            for metric in metrics:
                history[f"{metric.__name__}"] = running_avg_dict[f"{metric.__name__}_avg"]()
        ####################################

        return history

    def evaluate_single_epoch(self, model, val_dataloader, loss_fn, metrics=None, callbacks=None, device=None):

        model.eval()
        progress_bar = ProgressBar(total=len(val_dataloader), prefix="valid")
        val_loss_avg = RunningAverage()
        running_avg_dict = dict()
        history = dict()
        prefix = "val_"

        if metrics is not None:
            for metric in metrics:
                running_avg_dict[f"{prefix}{metric.__name__}_avg"] = RunningAverage()

        self.execute_callbacks(self, callbacks, "val_begin")
        for batch_ix, (X_val, y_val) in enumerate(val_dataloader):
            self.execute_callbacks(self, callbacks, "val_batch_begin")
            X_val = X_val.to(device)
            y_val = y_val.to(device)
            with torch.inference_mode():
                y_pred_val = model(X_val)
                batch_val_loss = loss_fn(y_pred_val, y_val)
            # update value + message
            val_loss_avg.update(batch_val_loss.item())
            message = f"{prefix}loss: {round(val_loss_avg(), 5)}"
            if metrics is not None:
                for metric in metrics:
                    running_avg_dict[f"{prefix}{metric.__name__}_avg"].update(metric(y_pred_val, y_val).item())
                    metric_value = round(running_avg_dict[f"{prefix}{metric.__name__}_avg"](), 5)
                    message += f" | {prefix}{metric.__name__}: {metric_value}"
            ################################################
            self.execute_callbacks(self, callbacks, "val_batch_end")
            progress_bar.update(batch_ix + 1, message)
        self.execute_callbacks(self, callbacks, "val_end")

        # update history
        history[f"{prefix}loss"] = val_loss_avg()
        if metrics is not None:
            for metric in metrics:
                history[f"{prefix}{metric.__name__}"] = running_avg_dict[f"{prefix}{metric.__name__}_avg"]()
        ####################################

        return history
