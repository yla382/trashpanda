"""The made-up trace files the tests of the viewer use. They are test data: they live here, not in the package."""

from pathlib import Path

DATA = Path(__file__).parent / "data"
SIMPLE = DATA / "dummy_trace.json"                 # one pass, a load phase, copies one block at a time
STEPS = DATA / "dummy_trace_steps.json"            # overlap=True, four sampling steps, a look-ahead of two
