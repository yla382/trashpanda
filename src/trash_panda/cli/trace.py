"""`trash-panda trace`: open a recorded run in the browser, list the recorded runs, or remove old ones."""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

from .. import paths


def _models(path):
    """(the names of the models, the duration in seconds) of a trace file, or None if it cannot be read as one."""
    import json

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        plan = data["trashPanda"]
        names = ", ".join(m["name"] for m in plan.get("models", [])) or "-"
        return names, plan["duration_us"] / 1e6
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _list(directory) -> None:
    from rich.console import Console
    from rich.table import Table

    files = paths.trace_files(directory)
    console = Console()
    if not files:
        console.print(f"No trace files in {directory}. Record one with TRASH_PANDA_TRACE=1.", markup=False)
        return
    table = Table(title=str(directory), title_justify="left")
    for name, justify in (("File", "left"), ("When", "left"), ("Size", "right"), ("Duration", "right"), ("Models", "left")):
        table.add_column(name, justify=justify, no_wrap=name != "Models", overflow="fold")
    for path in files:
        info = path.stat()
        read = _models(path)
        table.add_row(path.name, datetime.fromtimestamp(info.st_mtime).strftime("%Y-%m-%d %H:%M"), paths.fmt_size(info.st_size),
                      f"{read[1]:.1f} s" if read else "-", read[0] if read else "invalid")
    console.print(table)


def _clean(directory, keep, older_than, yes) -> None:
    files = paths.trace_files(directory)
    try:
        age = paths.parse_age(older_than) if older_than is not None else None
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2)
    if keep is not None and keep < 0:
        typer.echo("--keep must be 0 or more.", err=True)
        raise typer.Exit(2)
    doomed = paths.to_clean(files, keep=keep, older_than=age, now=time.time())
    if not doomed:
        typer.echo(f"Nothing to remove in {directory}.")
        return
    total = sum(p.stat().st_size for p in doomed)
    for path in doomed:
        typer.echo(f"  {path.name}   {paths.fmt_size(path.stat().st_size)}")
    if not yes and not typer.confirm(f"Remove these {len(doomed)} files ({paths.fmt_size(total)}) from {directory}? This cannot be undone"):
        typer.echo("Nothing removed.")
        raise typer.Exit(1)
    for path in doomed:
        path.unlink()
    typer.echo(f"Removed {len(doomed)} files ({paths.fmt_size(total)}).")


def trace_command(
    trace: Optional[Path] = typer.Argument(None, exists=True, readable=True, dir_okay=False,
                                           help="A trace file written by trash-panda. Without one, the newest in the trace folder."),
    port: int = typer.Option(0, "--port", help="The port to serve on. 0 picks a free one."),
    no_open: bool = typer.Option(False, "--no-open", help="Do not open the browser; only print the address."),
    list_files: bool = typer.Option(False, "--list", help="List the trace files in the trace folder and stop."),
    clean: bool = typer.Option(False, "--clean", help="Remove trace files from the trace folder (it asks first)."),
    keep: Optional[int] = typer.Option(None, "--keep", help="With --clean: keep the newest N files and remove the others."),
    older_than: Optional[str] = typer.Option(None, "--older-than", help="With --clean: remove files older than this, such as 30d, 12h or 2w."),
    yes: bool = typer.Option(False, "--yes", help="With --clean: do not ask first."),
) -> None:
    """Show a recorded run as a replay in the browser: blocks moving between disk, RAM and the GPU, and a timeline.

    Recording is switched on with TRASH_PANDA_TRACE=1; the files go to TRASH_PANDA_TRACE_DIR (default: the trace folder in
    ~/.cache/trash_panda). A small server runs on this machine only, until you press Ctrl+C.
    """
    directory = paths.trace_dir()
    if (list_files or clean) and trace is not None:
        typer.echo("--list and --clean work on the trace folder: give no file.", err=True)
        raise typer.Exit(2)
    if list_files and clean:
        typer.echo("Use --list or --clean, not both.", err=True)
        raise typer.Exit(2)
    if (keep is not None or older_than is not None or yes) and not clean:
        typer.echo("--keep, --older-than and --yes belong to --clean.", err=True)
        raise typer.Exit(2)
    if list_files:
        _list(directory)
        return
    if clean:
        _clean(directory, keep, older_than, yes)
        return

    from ..ui.load import TraceError, load_trace
    from ..ui.server import serve

    if trace is None:
        trace = paths.latest_trace(directory)
        if trace is None:
            typer.echo(f"No trace files in {directory}. Record one with TRASH_PANDA_TRACE=1, or give a file.", err=True)
            raise typer.Exit(1)
    try:
        load_trace(trace)                 # a bad file fails here, not as an empty page
    except TraceError as exc:
        typer.echo(f"{trace.name}: {exc}", err=True)
        raise typer.Exit(1)
    serve(trace, port=port, open_browser=not no_open, echo=typer.echo, directory=directory)
