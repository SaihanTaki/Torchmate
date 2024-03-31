<div align="center">
 
![logo](https://i.ibb.co/tzsKgkR/Torchmate-logo-v1.png)  
**A High level PyTorch Training and Utility Library**

[![GitHub Workflow Status (branch)](https://github.com/SaihanTaki/Torchmate/actions/workflows/pypi.yml/badge.svg)](https://github.com/SaihanTaki/Torchmate/actions/workflows/pypi.yml)
[![GitHub Workflow Status (branch)](https://github.com/SaihanTaki/Torchmate/actions/workflows/tests.yml/badge.svg)](https://github.com/SaihanTaki/Torchmate/actions/workflows/tests.yml) 
[![Codecov](https://img.shields.io/codecov/c/github/SaihanTaki/Torchmate?logo=codecov&labelColor=3B434B)](https://app.codecov.io/gh/SaihanTaki/Torchmate)
[![Read the Docs](https://img.shields.io/readthedocs/torchmate?style=flat&logo=readthedocs&logoColor=orange&color=blue&labelColor=3B434B)](https://torchmate.readthedocs.io/en/latest/)

[![PyPI](https://img.shields.io/pypi/v/torchmate?style=flat&labelColor=3B434B)](https://pypi.org/project/torchmate/)
[![Code Style](https://img.shields.io/badge/codestyle-black-black?style=flat&label=code%20style&labelColor=3B434B)](https://github.com/psf/black)
[![MIT License](https://img.shields.io/badge/License-MIT-<COLOR>.svg?style=flat&labelColor=3B434B)](https://github.com/SaihanTaki/Torchmate/blob/main/LICENSE.txt)
[![Python Version](https://img.shields.io/pypi/pyversions/torchmate?style=flat&labelColor=3B434B)](https://www.python.org/)
 

</div>

### [📚 Project Documentation](https://torchmate.readthedocs.io/en/latest/)

Visit [Torchmate's Read The Docs Project Page](https://torchmate.readthedocs.io/en/latest/) or read following README to know more about Torchmate library


### 💡 Introduction

So, why did I write Torchmate? I was a big fan of TensorFlow and Keras. But during my undergrad thesis I needed to use PyTorch. I was astonished with PyToroch’s flexibility. But I had to write the same boilerplate code which was quite frustrating to me. So, I decided to use a high level library like Catalyst or Lightning. Catalyst was great, but I missed Keras's verbose training output (which is cleaner) for better visualization (I know it's not a very good reason for writing a library). Pytorch Lightning also very good, but it changes a lot how we usually structure our code. Additionally, I was curious about how high-level frameworks like Keras, Catalyst, or Lightning work internally and utilize callbacks for extending functionalities. Building a minimalistic library myself seemed like the best way to understand these concepts. So, that's why I built Torchmate. Torchmate incorporates everything (actually not everything, some functionalities are still under development) I need and the way I prefere as a deep learning practitioner. 


### 🔑 Key Features

- **Encapsulate all training essentials:** Model, data loaders, loss function, optimizer, and learning rate schedulers.
- **Mixed precision training (AMP):** Train faster and potentially achieve better generalization with mixed precision calculations.
- **Gradient Accumulation:** Train on larger batches virtually by accumulating gradients, improving memory efficiency (Implemented through callback).
- **Gradient Clipping:** Prevent exploding gradients and stabilize training (Implemented through callback).
- **Gradient Penalty:** Enhance stability in generative models like GANs.
- **Callback Mechanism:** Monitor progress, save checkpoints, early stopping and extend functionality with custom callbacks.
- **Experiment Tracking:** Integrate dedicated tools like Weights & Biases or TensorBoard through callbacks.
- **Minimal Dependency:** Torchmate only requires four dependencies-  PyTorch (of course), NumPy, Matplotlib, and Weights & Biases (Wandb).
- **Research Paper Modules:** Implementations of various modules from research papers, facilitating reproducibility and experimentation.


### ⚙️ Installation

First make sure that PyTorch is installed on your envioronment. Then install torchmate via pip.

**From PyPI:**
```Bash
$ pip install torchmate
```
**From Source:**
```Bash
$ pip install git+https://github.com/SaihanTaki/torchmate
```

<br>

**N.B.** Torchmate requires the PyTorch (Of course) library to function. But torchmate does not list PyTorch as a dependency to avoid unnecessary overhead during installation. Excluding PyTorch as a dependency allows users to explicitly install the version of PyTorch best suited for their specific needs and environment. For instance, users who don't require GPU acceleration can install the CPU-only version of PyTorch, reducing unnecessary dependencies and installation size.PyTorch Installation Page: https://pytorch.org/get-started/locally/


### ⏳ Quick Example

```python
import torch
import numpy as np
from torchmate.trainer import Trainer
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

model = SimpleModel()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# Create a Trainer instance
trainer = Trainer(
    model,
    train_dataloader,
    val_dataloader,
    loss_fn,
    optimizer,
    num_epochs=3,
    device=device
)


# Train the model
history = trainer.fit()

```



### 🛡️ License <a name="license"></a>
Torchmate is distributed under [MIT License](https://github.com/SaihanTaki/torchmate/blob/master/LICENSE.txt)