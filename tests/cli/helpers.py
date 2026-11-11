"""What the CLI tests share: run a command, and read one row of the table it printed."""

import re

from typer.testing import CliRunner

from trash_panda.cli.app import app


# Typer draws its help in color whenever GITHUB_ACTIONS is set, which puts escape codes inside words such as "--min-free-ram".
_ANSI = re.compile(r"\[[0-9;?]*[A-Za-z]")


def plain(output):
    return _ANSI.sub("", output)


def run(*args):
    result = CliRunner().invoke(app, [str(a) for a in args])
    return result, " ".join(plain(result.output).split())     # the output with the colors, line breaks and table padding removed


def cells(result, first):
    """The cells of the table row whose first cell is `first` (whatever box characters the table is drawn with)."""
    for line in plain(result.output).splitlines():
        parts = [c.strip() for c in re.split(r"[│|]", line)[1:-1]]
        if parts and parts[0] == first:
            return parts
    raise AssertionError(f"no table row starts with {first!r}: {plain(result.output)}")
