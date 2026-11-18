"""`trash-panda config`: every setting and environment variable, with the value it has now."""

import os
from dataclasses import fields

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .. import paths

console = Console()


def _show(value) -> str:
    return str(value).lower() if isinstance(value, bool) else str(value)


def rows() -> list:
    """(variable, value, state) for every variable, the folders first. The state is "default" (not set), "set" or "invalid"."""
    from ..offload import config

    out = []
    for var, value in (("TRASH_PANDA_HOME", paths.cache_dir()), ("TRASH_PANDA_TRACE_DIR", paths.trace_dir())):
        out.append((var, str(value), "set" if os.environ.get(var, "").strip() else "default"))
    for field in fields(config.DEFAULTS):
        var = config.ENV_PREFIX + field.name.upper()
        default = getattr(config.DEFAULTS, field.name)
        raw = os.environ.get(var)
        if not raw:                                     # an empty variable counts as not set, as in config.current()
            out.append((var, _show(default), "default"))
            continue
        try:
            out.append((var, _show(config._parse(field.name, default, raw)), "set"))
        except ValueError:
            out.append((var, raw, "invalid"))
    return out


def _unknown(known) -> list:
    """TRASH_PANDA_ variables in the environment that nothing reads, which is usually a typo."""
    return sorted(name for name in os.environ if name.upper().startswith("TRASH_PANDA_") and name.upper() not in known)


def config_command() -> None:
    """Show every setting and environment variable with its current value. A value nobody has set is marked (default)."""
    shown = rows()
    table = Table()
    table.add_column("Variable", no_wrap=True)
    table.add_column("Value", overflow="fold")
    for var, value, state in shown:
        cell = Text(value)
        if state == "default":
            cell.append(" (default)", style="dim")
        elif state == "set":
            cell.stylize("bold")
        else:
            cell.append(" (invalid)", style="red")
        table.add_row(var, cell)
    console.print(table)

    unknown = _unknown({var for var, _, _ in shown})
    if unknown:
        console.print("Not a trash-panda variable, so it is ignored: " + ", ".join(unknown), markup=False)
    if any(state == "invalid" for _, _, state in shown):
        console.print("A value marked invalid cannot be read, and the library raises an error when it runs.", markup=False)
        raise typer.Exit(1)
