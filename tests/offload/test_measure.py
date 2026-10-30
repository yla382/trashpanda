from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from trash_panda.offload.measure import _peak_since, _start_peak, measure_activation_vram  # noqa: E402


@pytest.fixture
def cuda(monkeypatch):
    """Stand-in for the torch.cuda calls, so the measurement can run without a GPU. `state` is what the fake GPU reports."""
    state = SimpleNamespace(allocated=100, peak=400, events=[])
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: state.events.append("sync"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device=None: state.events.append("reset"))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: state.allocated)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: state.peak)
    return state


def test_start_peak_resets_the_counter_and_returns_the_allocated_memory(cuda):
    assert _start_peak() == 100
    assert cuda.events == ["sync", "reset"]


def test_the_peak_is_taken_beyond_what_was_allocated_before(cuda):
    assert _peak_since(100) == 300


def test_the_streaming_buffer_of_the_plan_is_subtracted(cuda):
    plan = SimpleNamespace(stream_buffer_bytes=50, staging_bytes=0)
    assert _peak_since(100, plan) == 250


def test_the_blocks_staged_ahead_are_subtracted_too(cuda):
    plan = SimpleNamespace(stream_buffer_bytes=50, staging_bytes=80)
    assert _peak_since(100, plan) == 170


def test_the_result_is_never_negative(cuda):
    plan = SimpleNamespace(stream_buffer_bytes=10 ** 6, staging_bytes=10 ** 6)
    assert _peak_since(100, plan) == 0


def test_measure_resets_before_the_call_and_runs_it_once(cuda):
    def fn():
        cuda.events.append("fn")

    assert measure_activation_vram(fn, SimpleNamespace(stream_buffer_bytes=50, staging_bytes=0)) == 250
    assert cuda.events.index("reset") < cuda.events.index("fn")
    assert cuda.events.count("fn") == 1
