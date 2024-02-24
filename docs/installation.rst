⚙️ Installation 
=================

From PyPI:

.. code-block:: bash

    $ pip install torchmate

From Source:

.. code-block:: bash

    $ pip install git+https://github.com/SaihanTaki/torchmate
    


**N.B.** Torchmate requires the PyTorch (Of course) library to function.
But torchmate does not list PyTorch as a dependency to avoid unnecessary overhead during installation.
Excluding PyTorch as a dependency allows users to explicitly install the version of PyTorch best
suited for their specific needs and environment. For instance, users who don't require GPU acceleration
can install the CPU-only version of PyTorch, reducing unnecessary dependencies and installation size.
PyTorch Installation Page: https://pytorch.org/get-started/locally/