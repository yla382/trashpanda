import contextlib
from concurrent.futures import Future

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file  # noqa: E402

from .tiny_model import Tiny, meta_model  # noqa: E402
from trash_panda.offload import (  # noqa: E402
    Plan,
    install_offload,
    offload_blocks,
    quantize_fp8,
    stream_load,
)
from trash_panda.offload import hooks  # noqa: E402
from trash_panda.offload.formats import expand  # noqa: E402

needs_fp8 = pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="this torch has no float8")
ALL = [f"blocks.{i}." for i in range(4)]


@pytest.mark.parametrize(
    "tiers",
    [
        dict(resident=ALL),
        dict(streamed=ALL),
        dict(disk=ALL),
        dict(resident=ALL[:1], streamed=ALL[1:3], disk=ALL[3:]),
    ],
    ids=["all-gpu", "all-ram", "all-disk", "mixed"],
)
def test_outputs_match_the_reference_model(tiny_checkpoint, tiers):
    ref, path = tiny_checkpoint
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", **tiers), device="cpu")
    handle = install_offload(model)
    x = torch.randn(3, 16)
    assert torch.allclose(model(x), ref(x), atol=1e-6)
    assert torch.allclose(model(x), ref(x), atol=1e-6)        # a second pass works too
    handle.remove()


def test_low_level_api_without_a_plan(tiny_checkpoint):
    ref, path = tiny_checkpoint
    model = meta_model()
    stream_load(model, path, offload_prefixes=["blocks."], compute_dtype="fp32", device="cpu")
    handle = offload_blocks(model.blocks, device="cpu", compute_dtype=torch.float32)
    x = torch.randn(2, 16)
    assert torch.allclose(model(x), ref(x), atol=1e-6)
    handle.remove()


def test_failed_forward_puts_the_weights_back(tiny_checkpoint):
    _, path = tiny_checkpoint
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", disk=ALL), device="cpu")
    install_offload(model)
    with pytest.raises(RuntimeError):
        model.blocks[0](torch.randn(3, 99))                    # wrong width: fails inside the block
    assert model.blocks[0].lin1.weight.is_meta                 # the compact (here: on-disk) placeholder is back


def test_remove_restores_the_blocks(tiny_checkpoint):
    _, path = tiny_checkpoint
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", disk=ALL), device="cpu")
    handle = install_offload(model)
    assert "forward" in model.blocks[0].__dict__
    handle.remove()
    assert "forward" not in model.blocks[0].__dict__


def test_report_lists_unexpected_keys(tiny_checkpoint, tmp_path):
    ref, _ = tiny_checkpoint
    state = {k: v.contiguous() for k, v in ref.state_dict().items()}
    state["extra.weight"] = torch.zeros(2, 2)
    path = tmp_path / "extra.safetensors"
    save_file(state, str(path))
    report = stream_load(meta_model(), path, compute_dtype="fp32", device="cpu")
    assert report.unexpected == ["extra.weight"]
    assert "extra.weight" in str(report)


def test_missing_tensors_are_an_error(tiny_checkpoint, tmp_path):
    ref, _ = tiny_checkpoint
    state = {k: v.contiguous() for k, v in ref.state_dict().items() if k != "head.bias"}
    path = tmp_path / "missing.safetensors"
    save_file(state, str(path))
    with pytest.raises(RuntimeError, match="head.bias"):
        stream_load(meta_model(), path, compute_dtype="fp32", device="cpu")


def test_install_offload_needs_a_plan(tiny_checkpoint):
    _, path = tiny_checkpoint
    model = meta_model()
    stream_load(model, path, offload_prefixes=["blocks."], compute_dtype="fp32", device="cpu")
    with pytest.raises(ValueError, match="needs a Plan"):
        install_offload(model)


def _fp8_checkpoint(tmp_path, with_scale):
    """A checkpoint whose block weights are fp8, plus the matching reference model (dequantized weights)."""
    torch.manual_seed(1)
    ref = Tiny().eval()
    tensors = {}
    for k, v in ref.state_dict().items():
        if k.startswith("blocks.") and k.endswith(".weight") and v.dim() == 2:
            q, scale = quantize_fp8(v)
            tensors[k] = q
            if with_scale:
                tensors[k + "_scale"] = scale
            v.copy_(expand(q, scale if with_scale else None, torch.float32, "cpu"))
        else:
            tensors[k] = v.clone()
    path = tmp_path / ("fp8_scaled.safetensors" if with_scale else "fp8_plain.safetensors")
    try:
        save_file(tensors, str(path))
    except Exception as exc:                                    # pragma: no cover
        pytest.skip(f"this safetensors cannot save float8: {exc}")
    return ref, path


@needs_fp8
def test_scaled_fp8_blocks_stay_compact_and_match(tmp_path):
    ref, path = _fp8_checkpoint(tmp_path, with_scale=True)
    model = meta_model()
    report = stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", streamed=ALL), device="cpu")
    assert report.kinds["fp8_scaled"] == 8                      # 4 blocks x 2 linear weights
    assert model.blocks[0].lin1.weight.dtype == torch.float8_e4m3fn
    handle = install_offload(model)
    x = torch.randn(3, 16)
    assert torch.allclose(model(x), ref(x), atol=1e-5)
    assert model.blocks[0].lin1.weight.dtype == torch.float8_e4m3fn    # back to compact form afterwards
    handle.remove()


@needs_fp8
def test_plain_fp8_without_scale_is_supported_and_flagged(tmp_path):
    ref, path = _fp8_checkpoint(tmp_path, with_scale=False)
    model = meta_model()
    report = stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", streamed=ALL), device="cpu")
    assert report.kinds["fp8_plain"] == 8
    assert any("no scale" in w for w in report.warnings)
    assert model.blocks[0].lin1.weight.dtype == torch.float8_e4m3fn   # still 1 byte per weight in RAM


@needs_fp8
def test_fp8_blocks_can_stream_from_disk(tmp_path):
    ref, path = _fp8_checkpoint(tmp_path, with_scale=True)
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", disk=ALL), device="cpu")
    install_offload(model)
    x = torch.randn(3, 16)
    assert torch.allclose(model(x), ref(x), atol=1e-5)


@needs_fp8
def test_quantize_on_load_makes_big_weights_fp8(tmp_path):
    torch.manual_seed(2)
    ref = Tiny(d=128, n=2).eval()
    path = tmp_path / "big.safetensors"
    save_file({k: v.contiguous() for k, v in ref.state_dict().items()}, str(path))
    model = meta_model(d=128, n=2)
    plan = Plan.manual(compute_dtype="fp32", storage_dtype="fp8", streamed=["blocks.0.", "blocks.1."])
    report = stream_load(model, path, plan=plan, device="cpu")
    assert report.kinds["fp8_scaled"] == 4                      # 2 blocks x 2 linear weights
    assert model.blocks[0].lin1.weight.dtype == torch.float8_e4m3fn
    assert model.blocks[0].norm.weight.dtype == torch.float32   # small tensors are left alone
    install_offload(model)
    x = torch.randn(3, 128)
    out, expected = model(x), ref(x)
    assert (out - expected).abs().max() < 0.2 * expected.abs().max()   # lossy, but close


def test_prefetcher_close_is_safe_without_cuda_and_can_repeat():
    prefetcher = hooks._Prefetcher(torch.device("cpu"))
    assert prefetcher.stream is None and prefetcher.worker is None and prefetcher.pool is None
    prefetcher.close()
    prefetcher.close()                                          # closing twice does nothing
    assert prefetcher.stream is None and prefetcher.worker is None and prefetcher.pool is None


def test_remove_closes_the_shared_prefetcher_once_and_last(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", streamed=ALL), device="cpu")
    closed = []

    def fake_close(self):
        closed.append("forward" in model.blocks[0].__dict__)    # was the block still wrapped when close ran?

    monkeypatch.setattr(hooks._Prefetcher, "close", fake_close)
    handle = install_offload(model)
    handle.remove()
    assert closed == [False]                                    # closed exactly once, after the blocks were restored


def test_offload_blocks_closes_only_a_prefetcher_it_created(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    closed = []
    monkeypatch.setattr(hooks._Prefetcher, "close", lambda self: closed.append(self))

    own = meta_model()
    stream_load(own, path, offload_prefixes=["blocks."], compute_dtype="fp32", device="cpu")
    offload_blocks(own.blocks, device="cpu", compute_dtype=torch.float32).remove()
    assert len(closed) == 1                                     # it made its own prefetcher, so it closes it

    closed.clear()
    shared = meta_model()
    stream_load(shared, path, offload_prefixes=["blocks."], compute_dtype="fp32", device="cpu")
    given = hooks._Prefetcher(torch.device("cpu"))
    offload_blocks(shared.blocks, device="cpu", compute_dtype=torch.float32, prefetcher=given).remove()
    assert closed == []                                         # a prefetcher passed in is left to its owner


def test_disk_reader_specs_describe_the_blocks_tensors(tiny_checkpoint):
    ref, path = tiny_checkpoint
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", disk=ALL), device="cpu")
    state = model._trash_panda_state
    reader = hooks.DiskReader(state.checkpoint, state.disk_keys, "blocks.1.")
    specs = reader.specs()
    expected = {k[len("blocks.1."):]: v for k, v in ref.state_dict().items() if k.startswith("blocks.1.")}
    assert {name for name, _kind, _info in specs} == set(expected)
    for name, kind, info in specs:
        assert kind == 0                                        # this checkpoint has no scale tensors
        assert tuple(info.shape) == tuple(expected[name].shape)
        assert info.torch_dtype == expected[name].dtype
        assert info.nbytes == expected[name].numel() * expected[name].element_size()


@needs_fp8
def test_disk_reader_specs_list_the_scales_of_fp8_weights(tmp_path):
    ref, path = _fp8_checkpoint(tmp_path, with_scale=True)
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", disk=ALL), device="cpu")
    state = model._trash_panda_state
    specs = hooks.DiskReader(state.checkpoint, state.disk_keys, "blocks.0.").specs()
    weights = [name for name, kind, _info in specs if kind == 0]
    scales = [name for name, kind, _info in specs if kind == 1]
    assert sorted(scales) == sorted(n for n in weights if n.endswith(".weight") and n.startswith("lin"))


class _FakeEvent:
    def record(self, stream=None):
        pass


@pytest.fixture
def cpu_prefetcher(monkeypatch):
    """A prefetcher whose CUDA calls are replaced by harmless stand-ins, so its staging jobs can run on the CPU."""
    monkeypatch.setattr(torch.cuda, "set_device", lambda index: None)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(hooks._PinnedPool, "acquire", lambda self, nbytes: (0, torch.empty(nbytes, dtype=torch.uint8)))
    prefetcher = object.__new__(hooks._Prefetcher)
    prefetcher.device, prefetcher.index, prefetcher.stream = torch.device("cpu"), 0, object()
    prefetcher.pool = hooks._PinnedPool(2)
    return prefetcher


def _disk_entry(path, prefix):
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", disk=ALL), device="cpu")
    state = model._trash_panda_state
    return hooks._Entry([], {}, hooks.DiskReader(state.checkpoint, state.disk_keys, prefix))


def test_disk_job_reads_a_blocks_tensors_through_the_buffer(tiny_checkpoint, cpu_prefetcher):
    ref, path = tiny_checkpoint
    tensors, _event = cpu_prefetcher._stage_disk_job(_disk_entry(path, "blocks.2."))
    expected = {k[len("blocks.2."):]: v for k, v in ref.state_dict().items() if k.startswith("blocks.2.")}
    assert set(tensors) == set(expected)
    for name, (weight, scale) in tensors.items():
        assert scale is None
        assert torch.equal(weight, expected[name])


def test_disk_job_still_works_without_a_pinned_buffer(tiny_checkpoint, cpu_prefetcher, monkeypatch):
    ref, path = tiny_checkpoint
    monkeypatch.setattr(hooks._PinnedPool, "acquire", lambda self, nbytes: None)          # pinning failed
    tensors, _event = cpu_prefetcher._stage_disk_job(_disk_entry(path, "blocks.3."))
    for name, (weight, _scale) in tensors.items():
        assert torch.equal(weight, ref.state_dict()["blocks.3." + name])


@needs_fp8
def test_disk_job_returns_the_scales_of_fp8_weights(tmp_path, cpu_prefetcher):
    ref, path = _fp8_checkpoint(tmp_path, with_scale=True)
    tensors, _event = cpu_prefetcher._stage_disk_job(_disk_entry(path, "blocks.0."))
    weight, scale = tensors["lin1.weight"]
    assert weight.dtype == torch.float8_e4m3fn and scale is not None
    assert torch.allclose(expand(weight, scale, torch.float32, "cpu"), ref.state_dict()["blocks.0.lin1.weight"])
    assert tensors["norm.weight"][1] is None                    # tensors without a scale have none


def test_ram_job_copies_a_blocks_tensors_through_the_buffer(tiny_checkpoint, cpu_prefetcher):
    ref, path = tiny_checkpoint
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", streamed=ALL), device="cpu")
    block = model.blocks[1]
    slots = [(name, None, None, True, p) for name, p in block.named_parameters()]
    tensors, _event = cpu_prefetcher._stage_job(hooks._Entry(slots, {}, None))
    for name, (weight, scale) in tensors.items():
        assert scale is None
        assert torch.equal(weight, ref.state_dict()["blocks.1." + name])


class _RecordingWorker:
    """Stands in for the worker thread: remembers which blocks were submitted and leaves their futures pending."""

    def __init__(self):
        self.submitted = []

    def submit(self, _job, entry):
        self.submitted.append(entry)
        return Future()


def _queue_entry(name, on_cpu=True):
    """An entry whose weights wait in CPU memory (needs a copy) or are 'already on the GPU' (a meta tensor here)."""
    weight = torch.zeros(1) if on_cpu else torch.zeros(1, device="meta")
    return hooks._Entry([(name, None, None, True, weight)], {}, None)


def _chain(*entries):
    for here, after in zip(entries, entries[1:]):
        here.next = after
    return entries


def _queue(lookahead):
    prefetcher = object.__new__(hooks._Prefetcher)
    prefetcher.device, prefetcher.stream, prefetcher.lookahead = torch.device("cpu"), object(), lookahead
    prefetcher.window, prefetcher.worker = [], _RecordingWorker()
    return prefetcher


def test_stage_ahead_stages_the_next_blocks_in_order():
    a, b, c, d = _chain(*[_queue_entry(n) for n in "abcd"])
    queue = _queue(lookahead=2)
    queue.stage_ahead(a)
    assert queue.worker.submitted == [b, c]
    assert queue.window == [b, c]
    assert d.staged is None


def test_blocks_already_on_the_gpu_do_not_use_up_the_window():
    a, r1, r2, b, c = _chain(_queue_entry("a"), _queue_entry("r1", False), _queue_entry("r2", False),
                             _queue_entry("b"), _queue_entry("c"))
    queue = _queue(lookahead=2)
    queue.stage_ahead(a)
    assert queue.worker.submitted == [b, c]             # the resident blocks in between are skipped


def test_stage_ahead_does_not_stage_a_block_twice():
    a, b, c = _chain(*[_queue_entry(n) for n in "abc"])
    queue = _queue(lookahead=2)
    queue.stage_ahead(a)
    queue.stage_ahead(a)
    assert queue.worker.submitted == [b, c]


def test_a_changed_order_drops_the_blocks_that_are_no_longer_coming():
    a, b, c = _chain(*[_queue_entry(n) for n in "abc"])
    x, d = _chain(_queue_entry("x"), _queue_entry("d"))
    queue = _queue(lookahead=2)
    queue.stage_ahead(a)
    stale = [b.staged, c.staged]
    queue.stage_ahead(x)                                # the model ran in a different order than we had learned
    assert queue.window == [d]
    assert b.staged is None and c.staged is None
    assert all(future.cancelled() for future in stale)  # their jobs had not started, so they were cancelled


def test_a_ring_of_blocks_does_not_loop_forever():
    a, b, c = _chain(*[_queue_entry(n) for n in "abc"])
    c.next = a                                          # the order wraps around to the first block
    queue = _queue(lookahead=5)
    queue.stage_ahead(a)
    assert queue.worker.submitted == [b, c]             # it stops when it comes back to the current block


def test_lookahead_zero_stages_nothing():
    a, b = _chain(*[_queue_entry(n) for n in "ab"])
    queue = _queue(lookahead=0)
    queue.stage_ahead(a)
    assert queue.worker.submitted == [] and queue.window == []


def test_lookahead_reaches_the_prefetcher(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    seen = []
    original_init = hooks._Prefetcher.__init__

    def recording_init(self, device, lookahead=2, profile=None):
        seen.append(lookahead)
        original_init(self, device, lookahead, profile)

    monkeypatch.setattr(hooks._Prefetcher, "__init__", recording_init)

    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", streamed=ALL), device="cpu")
    install_offload(model).remove()                                          # the default
    install_offload(model, lookahead=5).remove()
    offload_blocks(model.blocks, device="cpu", compute_dtype=torch.float32, lookahead=3).remove()
    assert seen == [2, 5, 3]


@pytest.mark.parametrize("lookahead", [0, 1, 3])
def test_outputs_match_the_reference_for_any_lookahead(tiny_checkpoint, lookahead):
    ref, path = tiny_checkpoint
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", resident=ALL[:1], streamed=ALL[1:3], disk=ALL[3:]),
                device="cpu")
    handle = install_offload(model, lookahead=lookahead)
    x = torch.randn(3, 16)
    assert torch.allclose(model(x), ref(x), atol=1e-6)
    assert torch.allclose(model(x), ref(x), atol=1e-6)                        # and again, once the order is learned
    handle.remove()


class _SyncEvent:
    """A stand-in for a CUDA event that remembers whether the host waited for it."""

    def __init__(self):
        self.synchronized = False

    def record(self, stream=None):
        pass

    def synchronize(self):
        self.synchronized = True


class _FakeStream:
    def synchronize(self):
        pass


def _throttled(lookahead, monkeypatch):
    monkeypatch.setattr(torch.cuda, "Event", _SyncEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: object())
    prefetcher = object.__new__(hooks._Prefetcher)
    prefetcher.device, prefetcher.stream, prefetcher.lookahead = torch.device("cpu"), _FakeStream(), lookahead
    prefetcher.in_flight = hooks.deque()
    return prefetcher


def test_the_gpu_is_not_allowed_to_fall_more_than_two_blocks_behind(monkeypatch):
    prefetcher = _throttled(2, monkeypatch)
    events = []
    for _ in range(4):
        prefetcher.wait_for_room()                       # what runs before each block starts
        prefetcher.block_done()                          # and after it
        events.append(prefetcher.in_flight[-1])
        assert len(prefetcher.in_flight) <= 2            # never more than two blocks in flight
    assert [e.synchronized for e in events] == [True, True, False, False]   # the oldest ones were waited for


def test_waiting_for_room_does_nothing_when_prefetching_is_off(monkeypatch):
    prefetcher = _throttled(0, monkeypatch)
    for _ in range(5):
        prefetcher.wait_for_room()
        prefetcher.block_done()
    assert len(prefetcher.in_flight) == 0                # nothing recorded, nothing waited for


def test_close_forgets_the_blocks_in_flight(monkeypatch):
    prefetcher = _throttled(2, monkeypatch)
    prefetcher.block_done()
    prefetcher.worker, prefetcher.pool, prefetcher.window, prefetcher.last = None, None, [], None
    prefetcher.close()
    assert len(prefetcher.in_flight) == 0


def test_install_offload_follows_the_plans_lookahead_unless_told_otherwise(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    seen = []
    original_init = hooks._Prefetcher.__init__

    def recording_init(self, device, lookahead=2, profile=None):
        seen.append(lookahead)
        original_init(self, device, lookahead, profile)

    monkeypatch.setattr(hooks._Prefetcher, "__init__", recording_init)
    p = Plan.manual(compute_dtype="fp32", streamed=ALL)
    p.lookahead = 4
    model = meta_model()
    stream_load(model, path, plan=p, device="cpu")
    install_offload(model).remove()                       # follows the plan
    install_offload(model, lookahead=1).remove()          # an explicit value wins
    assert seen == [4, 1]


def test_profiling_is_off_unless_asked_for(monkeypatch):
    monkeypatch.delenv("TRASH_PANDA_PROFILE", raising=False)
    assert hooks._Prefetcher(torch.device("cpu")).profile is False
    monkeypatch.setenv("TRASH_PANDA_PROFILE", "1")
    assert hooks._Prefetcher(torch.device("cpu")).profile is True
    assert hooks._Prefetcher(torch.device("cpu"), profile=False).profile is False


def test_learn_counts_a_pass_each_time_the_first_block_comes_round():
    prefetcher = hooks._Prefetcher(torch.device("cpu"), profile=True)
    a, b = _chain(*[_queue_entry(n) for n in "ab"])
    for entry in (a, b, a, b, a):
        prefetcher.learn(entry)
    assert prefetcher.stats["passes"] == 3


def test_report_summarises_the_stats_once():
    prefetcher = hooks._Prefetcher(torch.device("cpu"), profile=True)
    prefetcher.stats.update(passes=2, disk_read_s=4.0, disk_read_gb=2.0, disk_blocks=10,
                            wait_staged_s=1.0, takes=8, takes_ready=6, wait_gpu_s=0.5)
    text = prefetcher.report()
    assert "disk reads: 1.00 GB in 2.00 s (500 MB/s), 5 blocks" in text
    assert "6 of 8 were ready at once" in text
    assert "waited for the GPU to catch up: 0.25 s" in text
    assert prefetcher.report() == ""                      # printed once only


def test_report_says_nothing_when_profiling_is_off():
    assert hooks._Prefetcher(torch.device("cpu"), profile=False).report() == ""


def test_learn_gives_each_block_a_position_in_the_pass():
    prefetcher = hooks._Prefetcher(torch.device("cpu"), profile=True)
    a, b, c = _chain(*[_queue_entry(n) for n in "abc"])
    for entry in (a, b, c, a, b, c):
        prefetcher.learn(entry)
    assert [prefetcher.where[id(e)] for e in (a, b, c)] == [0, 1, 2]


def test_report_lists_the_slowest_waits_by_block_position():
    prefetcher = hooks._Prefetcher(torch.device("cpu"), profile=True)
    prefetcher.stats.update(passes=2, takes=4, takes_ready=2, wait_staged_s=3.0)
    prefetcher.block_wait.update({0: 2.0, 7: 0.5, 3: 0.5})
    prefetcher.block_late.update({0: 2, 7: 1, 3: 0})
    text = prefetcher.report()
    assert "block 0: 1.00 s, late in 2 of 2" in text
    assert text.index("block 0") < text.index("block 7")


class _WaitingEvent:
    """A stand-in for a CUDA event whose synchronize() takes a known time on a fake clock."""

    def __init__(self, clock, seconds):
        self.clock, self.seconds = clock, seconds

    def synchronize(self):
        self.clock[0] += self.seconds


def test_each_pass_is_timed_and_its_waiting_recorded(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(hooks.time, "perf_counter", lambda: clock[0])
    prefetcher = hooks._Prefetcher(torch.device("cpu"), profile=True)
    a, b = _chain(*[_queue_entry(n) for n in "ab"])

    clock[0] = 10.0
    prefetcher.learn(a)                                   # pass 1 starts
    clock[0] = 11.0
    prefetcher.learn(b)
    prefetcher.stats["wait_staged_s"] += 2.0              # the main thread waited 2 s for staged blocks
    clock[0] = 14.0
    prefetcher.block_done()                               # pass 1 ends here
    clock[0] = 20.0
    prefetcher.learn(a)                                   # pass 2 starts, pass 1 is closed
    assert prefetcher.pass_log == [(4.0, 2.0)]
    prefetcher.stats["wait_staged_s"] += 0.5
    clock[0] = 23.0
    prefetcher.block_done()

    text = prefetcher.report()                            # closes pass 2
    assert prefetcher.pass_log == [(4.0, 2.0), (3.0, 0.5)]
    assert "pass 1: 4.00 s, waited 2.00 s" in text
    assert "pass 2: 3.00 s, waited 0.50 s" in text


def test_a_long_run_shows_only_the_first_and_last_passes(monkeypatch):
    prefetcher = hooks._Prefetcher(torch.device("cpu"), profile=True)
    prefetcher.stats["passes"] = 12
    prefetcher.pass_log = [(1.0 * n, 0.1 * n) for n in range(1, 13)]
    text = prefetcher.report()
    assert "pass 5:" in text and "pass 6:" not in text and "pass 11:" in text and "pass 12:" in text
    assert "    ..." in text


@pytest.mark.filterwarnings("ignore:trash-panda")
def test_the_wait_for_a_free_pinned_buffer_is_timed(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(hooks.time, "perf_counter", lambda: clock[0])
    pool = hooks._PinnedPool(2)
    pool.events[0] = _WaitingEvent(clock, 1.5)            # the buffer is still being read by the GPU
    pool.acquire(10)                                      # pinning fails on the CPU and returns None, after the wait
    assert pool.wait_s == 1.5
    assert pool.events[0] is None                         # the pool must survive the wait (it once overwrote self)
    pool.events[1] = _WaitingEvent(clock, 0.5)
    pool.acquire(10)
    assert pool.wait_s == 2.0


def test_the_report_shows_the_pinned_buffer_wait_per_pass():
    prefetcher = hooks._Prefetcher(torch.device("cpu"), profile=True)
    prefetcher.stats["passes"] = 2
    prefetcher.pool = hooks._PinnedPool(2)
    prefetcher.pool.wait_s = 3.0
    assert "worker waited for a free pinned buffer: 1.50 s" in prefetcher.report()


def test_the_report_has_no_buffer_line_without_a_pool():
    prefetcher = hooks._Prefetcher(torch.device("cpu"), profile=True)       # on the CPU there is no pool
    prefetcher.stats["passes"] = 1
    assert "pinned buffer" not in prefetcher.report()


def _refuse_pinned_memory(monkeypatch):
    """Make torch.empty(pin_memory=True) fail the way it does when pinned memory cannot be allocated."""
    real_empty = torch.empty

    def empty(*args, **kwargs):
        if kwargs.get("pin_memory"):
            raise RuntimeError("no pinned memory")
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)


def test_failing_to_pin_memory_warns_once_and_falls_back(monkeypatch):
    _refuse_pinned_memory(monkeypatch)
    pool = hooks._PinnedPool(2)
    with pytest.warns(RuntimeWarning, match="pinned") as seen:
        assert pool.acquire(10 ** 6) is None                  # the caller then copies through ordinary memory
        assert pool.acquire(10 ** 6) is None
        assert pool.acquire(10 ** 6) is None
    assert len(seen) == 1                                     # one warning per pool, not one per block
    assert "no pinned memory" in str(seen[0].message)
    assert "ordinary memory" in str(seen[0].message)


def test_no_warning_when_pinning_works(monkeypatch):
    real_empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, **k: real_empty(*a, **{**k, "pin_memory": False}))
    pool = hooks._PinnedPool(2)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")                        # any warning would fail the test
        index, buffer = pool.acquire(100)
    assert index == 0 and buffer.numel() == 100
