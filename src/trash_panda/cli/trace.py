"""`trash-panda trace`: open a recorded run in the browser."""

from pathlib import Path
from typing import Optional

import typer


def trace_command(
    trace: Optional[Path] = typer.Argument(None, exists=True, readable=True, dir_okay=False,
                                           help="A trace file. Without one, a made-up demo trace is shown."),
    port: int = typer.Option(0, "--port", help="The port to serve on. 0 picks a free one."),
    no_open: bool = typer.Option(False, "--no-open", help="Do not open the browser; only print the address."),
) -> None:
    """Show a recorded run as a replay in the browser: blocks moving between disk, RAM and the GPU, and a timeline.

    A small server runs on this machine only, until you press Ctrl+C.
    """
    from ..ui.load import DEMO, TraceError, load_trace
    from ..ui.server import serve

    path = trace or DEMO
    try:
        load_trace(path)                  # a bad file fails here, not as an empty page
    except TraceError as exc:
        typer.echo(f"{path.name}: {exc}", err=True)
        raise typer.Exit(1)
    if trace is None:
        typer.echo("No trace file given: showing the demo trace. It is made-up data, not a real run.")
    serve(path, port=port, open_browser=not no_open, echo=typer.echo)
