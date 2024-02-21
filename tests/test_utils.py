import sys

import pytest
import torch

sys.path.append("..")

from torchmate.utils import (  # noqa: E402
    HistoryPlotter,
    ProgressBar,
    RunningAverage,
    colorize_text,
)

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Cuda is not available")


# RunningAverage


def test_running_average_initialization():
    avg = RunningAverage()
    assert avg.steps == 0
    assert avg.total == 0


def test_running_average_update():
    avg = RunningAverage()
    avg.update(2)
    avg.update(4)
    assert avg.steps == 2
    assert avg.total == 6


def test_running_average_call():
    avg = RunningAverage()
    avg.update(2)
    avg.update(4)
    assert avg() == 3.0


def test_running_average_no_update():
    avg = RunningAverage()
    assert avg() == 0.0


# ProgressBar


@pytest.fixture
def progress_bar():
    return ProgressBar(total=100, bar_length=30, prefix="Testing")


def test_progress_bar_init(progress_bar):
    assert progress_bar.total == 100
    assert progress_bar.length == 30
    assert progress_bar.prefix == "Testing: "


def test_progress_bar_update(progress_bar, capsys):
    for i in range(100):
        progress_bar.update(i + 1, message=f"Processing item {i + 1}")
        captured = capsys.readouterr()
        assert captured.out.startswith("\rTesting")


def test_progress_bar_format_time():
    assert ProgressBar.format_time(0.5) == "500ms"
    assert ProgressBar.format_time(30) == "30.00s"
    assert ProgressBar.format_time(67.5) == "01:07min"
    assert ProgressBar.format_time(3663) == "01:01:03"


def test_progress_bar_complete(progress_bar, capsys):
    for i in range(100):
        progress_bar.update(i + 1, message=f"Processing item {i + 1}")
    captured = capsys.readouterr()
    assert captured.out.endswith("\n")


# colorize_text


def test_colorize_text_default():
    result = colorize_text("Test")
    assert result == "\033[38;2;0;0;0mTest\033[0m"


def test_colorize_text_foreground_only():
    result = colorize_text("Test", fore_tuple=(255, 0, 0))
    assert result == "\033[38;2;255;0;0mTest\033[0m"


def test_colorize_text_foreground_and_background():
    result = colorize_text("Test", fore_tuple=(255, 0, 0), back_tuple=(0, 255, 0))
    assert result == "\033[38;2;255;0;0;48;2;0;255;0mTest\033[0m"


def test_colorize_text_bold():
    result = colorize_text("Test", bold_text=True)
    assert result == "\033[1m\033[38;2;0;0;0mTest\033[0m"


def test_colorize_text_invalid_rgb():
    with pytest.raises(ValueError):
        colorize_text("Test", fore_tuple=(-1, 0, 0))

    with pytest.raises(ValueError):
        colorize_text("Test", back_tuple=(256, 0, 0))


def test_colorize_text_optional_parameters():
    result = colorize_text("Test", fore_tuple=(255, 0, 0), back_tuple=(0, 255, 0), bold_text=True)
    assert result == "\033[1m\033[38;2;255;0;0;48;2;0;255;0mTest\033[0m"


def test_colorize_text_reset():
    result = colorize_text("Test", fore_tuple=(255, 0, 0))
    assert result == "\033[38;2;255;0;0mTest\033[0m"


def test_colorize_text_bold_reset():
    result = colorize_text("Test", bold_text=True)
    assert result == "\033[1m\033[38;2;0;0;0mTest\033[0m"


# Testing the HistoryPlotter util

# This test needs PyQT5 or any other gui backend
# supported by matplotlib installed

# @pytest.fixture
# def example_history():
#     # Example history data
#     history = {
#         "loss": [0.5, 0.4, 0.3],
#         "val_loss": [0.6, 0.5, 0.4],
#         "lr": [0.001, 0.0001, 0.00001],
#         "accuracy": [0.8, 0.85, 0.9],
#         "val_accuracy": [0.75, 0.78, 0.82]
#     }
#     return history

# def test_history_plotter(example_history):
#     plotter = HistoryPlotter(history=example_history)

#     plotter.plot_all()
#     plotter.plot_loss()
#     plotter.plot_lr()
#     plotter.plot_metrics()
