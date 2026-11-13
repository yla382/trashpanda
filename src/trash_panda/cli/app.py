"""The `trash-panda` command: the app and its commands."""

try:
    import typer
except ImportError:
    raise SystemExit("The command line needs extra packages: pip install 'dumpsterpanda[cli]'")

from .info import info_command
from .plan import plan_command
from .trace import trace_command

app = typer.Typer(no_args_is_help=True, add_completion=False, help="trash-panda: plan and inspect offloaded models.")


@app.callback()
def _root() -> None:
    """The root of the commands: `trash-panda plan ...`, `trash-panda info ...`, `trash-panda trace ...`."""


app.command("plan")(plan_command)
app.command("info")(info_command)
app.command("trace")(trace_command)


def main() -> None:
    app()
