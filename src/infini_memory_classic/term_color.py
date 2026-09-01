from __future__ import annotations

"""
term_color: Unified terminal colorization utility.

Only colorizes text when `enabled=True`; otherwise returns it as-is.
"""

RESET = "\033[0m"
BOLD = "\033[1m"

FG = {
    "black": "\033[30m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
}

# Bright colors (high-intensity foreground)
BRIGHT_FG = {
    "bright_black": "\033[90m",
    "bright_red": "\033[91m",
    "bright_green": "\033[92m",
    "bright_yellow": "\033[93m",
    "bright_blue": "\033[94m",
    "bright_magenta": "\033[95m",
    "bright_cyan": "\033[96m",
    "bright_white": "\033[97m",
}


def _lookup_color(color: str | None) -> str | None:
    if not color:
        return None
    if color in FG:
        return FG[color]
    if color in BRIGHT_FG:
        return BRIGHT_FG[color]
    return None


def colorize(text: str, color: str | None = None, *, enabled: bool = True, bold: bool = False) -> str:
    """Wrap text with the given foreground color.

    - text: The text to colorize
    - color: Color name (e.g. 'red'/'yellow'/'cyan'); unknown names result in no colorization
    - enabled: When False, returns text as-is
    - bold: When True, applies bold (can be combined with color)
    """
    if not enabled:
        return text
    parts = []
    if bold:
        parts.append(BOLD)
    code = _lookup_color(color)
    if code:
        parts.append(code)
    if not parts:
        return text
    return f"{''.join(parts)}{text}{RESET}"


def red(text: str, *, enabled: bool = True, bold: bool = False) -> str:
    return colorize(text, "red", enabled=enabled, bold=bold)


def yellow(text: str, *, enabled: bool = True, bold: bool = False) -> str:
    return colorize(text, "yellow", enabled=enabled, bold=bold)


def cyan(text: str, *, enabled: bool = True, bold: bool = False) -> str:
    return colorize(text, "cyan", enabled=enabled, bold=bold)


def green(text: str, *, enabled: bool = True, bold: bool = False) -> str:
    return colorize(text, "green", enabled=enabled, bold=bold)


def blue(text: str, *, enabled: bool = True, bold: bool = False) -> str:
    return colorize(text, "blue", enabled=enabled, bold=bold)


def magenta(text: str, *, enabled: bool = True, bold: bool = False) -> str:
    return colorize(text, "magenta", enabled=enabled, bold=bold)


def white(text: str, *, enabled: bool = True, bold: bool = False) -> str:
    return colorize(text, "white", enabled=enabled, bold=bold)


def black(text: str, *, enabled: bool = True, bold: bool = False) -> str:
    return colorize(text, "black", enabled=enabled, bold=bold)


def bold(text: str, *, enabled: bool = True) -> str:
    return colorize(text, None, enabled=enabled, bold=True)


def format_git_diff(old_text: str, new_text: str, *, enabled: bool = True) -> str:
    """Generate a git-style colored diff output.

    Uses difflib to generate unified diff format with colors:
    - Red: removed lines (starting with -)
    - Green: added lines (starting with +)
    - Cyan: diff header info (starting with @@), and --- / +++ file path lines
    - No color: context lines (starting with a space)

    Style follows git diff unified diff format:
        --- a/old
        +++ b/new
        @@ -1,4 +1,5 @@
         context line
        -removed line
        +added line
    """
    import difflib

    lines_old = old_text.splitlines(keepends=True)
    lines_new = new_text.splitlines(keepends=True)

    # Generate unified diff
    diff = difflib.unified_diff(
        lines_old,
        lines_new,
        fromfile="a/old",
        tofile="b/new",
        lineterm=""
    )

    result_lines = []
    for line in diff:
        if line.startswith("---") or line.startswith("+++"):
            # File path lines: cyan
            result_lines.append(cyan(line, enabled=enabled))
        elif line.startswith("@@"):
            # Hunk header: cyan
            result_lines.append(cyan(line, enabled=enabled))
        elif line.startswith("-"):
            # Removed lines: red
            result_lines.append(red(line, enabled=enabled))
        elif line.startswith("+"):
            # Added lines: green
            result_lines.append(green(line, enabled=enabled))
        else:
            # Context lines: keep original color
            result_lines.append(line)

    return "".join(result_lines)


__all__ = [
    "colorize",
    "bold",
    "red",
    "yellow",
    "cyan",
    "green",
    "blue",
    "magenta",
    "white",
    "black",
    "format_git_diff",
]
