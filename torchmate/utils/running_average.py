class RunningAverage:
    """A simple class that maintains the running average of a quantity

    **Example usage:**

    .. code-block:: python

        loss_avg = RunningAverage()
        loss_avg.update(2)
        loss_avg.update(4)
        loss_avg() = 3  ## 2+4/2=3

    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val: int) -> None:
        self.total += val
        self.steps += 1

    def __call__(self):
        if self.steps:
            avg = self.total / (float(self.steps))
        else:
            avg = 0.0
        return avg
