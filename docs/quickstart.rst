🚀 Quick Start
===============


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
        