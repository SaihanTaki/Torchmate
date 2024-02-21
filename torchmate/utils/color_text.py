from typing import Optional, Tuple


def colorize_text(
    text: str,
    fore_tuple: Tuple[int, int, int] = (0, 0, 0),
    back_tuple: Optional[Tuple[int, int, int]] = None,
    bold_text: bool = False,
) -> str:
    """Return text formatted with specified foreground color, background color, and optional bold formatting.

    Args:
       text (str): The text to be colorized.
       fore_tuple (tuple, optional): A tuple of RGB values (0-255) for the foreground color. Defaults to ``(0, 0, 0)``.
       back_tuple (tuple, optional): A tuple of RGB values (0-255) for the background color. Defaults to ``(255, 255, 255)``.
       bold_text (bool, optional): Whether to apply bold formatting to the text. Defaults to ``False``.

    Raises:
       ValueError: If RGB values in fore_tuple or back_tuple are outside the valid range of 0-255.

    Returns:
       Str: The formatted text with ANSI color codes embedded.


    """

    if back_tuple is not None:
        for value in fore_tuple + back_tuple:
            if value not in range(0, 256):
                raise ValueError(f"Invalid RGB value: {value}. RGB values must be between 0 and 255.")
    else:
        for value in fore_tuple:
            if value not in range(0, 256):
                raise ValueError(f"Invalid RGB value: {value}. RGB values must be between 0 and 255.")

    bold_code = "\033[1m"
    reset_code = "\033[0m"
    start_command = "\033["
    change_fg = "38;2;"
    change_bg = "48;2;"
    end_command = "m"

    if back_tuple is not None:
        rf, gf, bf = fore_tuple
        rb, gb, bb = back_tuple
        foreground = change_fg + str(rf) + ";" + str(gf) + ";" + str(bf)
        background = change_bg + str(rb) + ";" + str(gb) + ";" + str(bb)
        ansi_color = start_command + foreground + ";" + background + end_command
    else:
        rf, gf, bf = fore_tuple
        foreground = change_fg + str(rf) + ";" + str(gf) + ";" + str(bf)
        ansi_color = start_command + foreground + end_command

    if bold_text:
        text = bold_code + ansi_color + text + reset_code
    else:
        text = ansi_color + text + reset_code

    return text


class COLOR:
    """
    BG = Background

    Reference:
        https://web.archive.org/web/20201214113226/http://ascii-table.com/ansi-escape-sequences.php
    """

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    PURPLE = "\033[95m"

    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    INVERSE = "\033[7m"
    END = "\033[0m"

    DEFAULT_BG = "\033[41m"
    BLACK_BG = "\033[40m"
    RED_BG = "\033[41m"
    GREEN_BG = "\033[42m"
    YELLOW_BG = "\033[43m"
    BLUE_BG = "\033[44m"
    MAGENTA_BG = "\033[45m"
    CYAN_BG = "\033[46m"
    WHITE_BG = "\033[47m"
