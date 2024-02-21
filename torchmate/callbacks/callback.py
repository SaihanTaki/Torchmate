class Callback:
    """Base class for creating callback objects in an experimental or training framework.

    Callbacks are used to customize and extend the behavior of an experiment, training loop,
    or optimization process by hooking into various stages of the execution.

    **Callback Methods:**

        ``on_experiment_begin(self, trainer: Trainer) -> None``
            Called at the beginning of an experiment.

        ``on_experiment_end(self, trainer: Trainer) -> None``
            Called at the end of an experiment.


        ``on_epoch_begin(self, trainer: Trainer) -> None``
            Called at the beginning of an epoch.

        ``on_epoch_begin(self, trainer: Trainer) -> None``
            Called at the end of an epoch.

        ``on_(train|val|predict)_begin(self, trainer: Trainer) -> None``
            Called at the beginning of fit/evaluate/predict.

        ``on_(train|val|predict)_end(self, trainer: Trainer) -> None``
            Called at the end of fit/evaluate/predict.

        ``on_(train|val|predict)_batch_begin(self, trainer: Trainer) -> None``
            Called right before processing a batch during training/validating/predicting.

        ``on_(train|val|predict)_batch_end(self, trainer: Trainer) -> None``
            Called at the end of training/validating/predicting a batch.

    Parameters:
        trainer (Trainer) -  An instance of (torchmate.trainer.Trainer) class.


    Note:
        This base class provides empty implementations for all callback methods, allowing derived callback classes
        to selectively override only the methods that need to be customized.


    **Example Usage:**
        Below is an example of a custom callback class that inherits from Callback and overrides
        specific methods to customize behavior during training:

    .. code-block:: python

        class CustomCallback(Callback):
            def __init__(self):
                self.current_epoch = 0

            def on_epoch_begin(self, trainer):
                self.current_epoch +=1
                print(f"Epoch {self.current_epoch} begins!")

            def on_epoch_end(self, trainer):
                print(f"Epoch {self.current_epoch} has finished!")

            def on_experiment_end(self, trainer):
                print("Experiment finished!")
                print(f"History: {trainer.history}")

        # Create an instance of the custom callback and use it during training
        custom_callback = CustomCallback()



    """

    def on_experiment_begin(self, trainer) -> None:
        pass

    def on_experiment_end(self, trainer) -> None:
        pass

    def on_epoch_begin(self, trainer) -> None:
        pass

    def on_epoch_end(self, trainer) -> None:
        pass

    def on_train_begin(self, trainer) -> None:
        pass

    def on_train_end(self, trainer) -> None:
        pass

    def on_train_batch_begin(self, trainer) -> None:
        pass

    def on_train_batch_end(self, trainer) -> None:
        pass

    def on_val_begin(self, trainer) -> None:
        pass

    def on_val_end(self, trainer) -> None:
        pass

    def on_val_batch_begin(self, trainer) -> None:
        pass

    def on_val_batch_end(self, trainer) -> None:
        pass

    def on_predict_begin(self, trainer) -> None:
        pass

    def on_predict_end(self, trainer) -> None:
        pass

    def on_predict_batch_begin(self, trainer) -> None:
        pass

    def on_predict_batch_end(self, trainer) -> None:
        pass

    def on_backward_end(self, trainer) -> None:
        pass
