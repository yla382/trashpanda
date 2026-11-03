import queue

import pytest

torch = pytest.importorskip("torch")

from .tiny_model import meta_model  # noqa: E402
from trash_panda.offload import Plan, install_offload, module_nbytes, stream_load  # noqa: E402
from trash_panda.offload.hooks import Release  # noqa: E402

ALL = [f"blocks.{i}." for i in range(4)]


def _loaded(path, dtype="fp32", **tiers):
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype=dtype, **tiers), device="cpu")
    return model, install_offload(model)


def _drain(q):
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


def test_subscribe_gives_a_queue_and_the_same_one_every_time(tiny_checkpoint):
    _, path = tiny_checkpoint
    _, handle = _loaded(path, resident=ALL)
    q = handle.subscribe()
    assert isinstance(q, queue.SimpleQueue)
    assert handle.subscribe() is q


def test_nothing_is_queued_before_the_final_pass(tiny_checkpoint):
    _, path = tiny_checkpoint
    model, handle = _loaded(path, resident=ALL[:2], streamed=ALL[2:], dtype="bf16")
    q = handle.subscribe()
    model(torch.randn(3, 16).bfloat16())
    model(torch.randn(3, 16).bfloat16())
    assert q.empty()


def test_nothing_is_announced_without_a_subscriber_or_the_profile(tiny_checkpoint):
    _, path = tiny_checkpoint
    model, handle = _loaded(path, streamed=ALL, dtype="bf16")
    handle._prefetcher.announce = lambda *a: pytest.fail("a block was announced with nobody watching")
    handle.final_pass()
    model(torch.randn(3, 16).bfloat16())
    assert all(p.is_meta for p in model.blocks.parameters())


def test_a_final_pass_queues_every_wrapped_ram_block_in_the_order_it_ran(tiny_checkpoint):
    _, path = tiny_checkpoint
    model, handle = _loaded(path, streamed=ALL, dtype="bf16")      # stored as fp32, so every block is wrapped
    stored = [module_nbytes(block) for block in model.blocks]
    q = handle.subscribe()
    handle.final_pass()
    model(torch.randn(3, 16).bfloat16())
    got = _drain(q)
    assert all(isinstance(r, Release) for r in got)
    assert [r.name for r in got] == [f"blocks.{i}" for i in range(4)]
    assert [r.tier for r in got] == ["ram"] * 4
    assert [r.nbytes for r in got] == stored
    assert all(r.event is None for r in got)                       # no CUDA here
    assert [r.queued for r in got] == sorted(r.queued for r in got)


def test_disk_blocks_are_queued_with_no_bytes(tiny_checkpoint):
    _, path = tiny_checkpoint
    model, handle = _loaded(path, disk=ALL, dtype="bf16")
    q = handle.subscribe()
    handle.final_pass()
    model(torch.randn(3, 16).bfloat16())
    got = _drain(q)
    assert [r.tier for r in got] == ["disk"] * 4
    assert [r.nbytes for r in got] == [0] * 4


def test_blocks_that_stay_on_the_gpu_are_queued_with_their_weights(tiny_checkpoint):
    _, path = tiny_checkpoint
    model, handle = _loaded(path, resident=ALL)
    held = [module_nbytes(block) for block in model.blocks]
    q = handle.subscribe()
    handle.final_pass()
    model(torch.randn(3, 16))
    got = _drain(q)
    assert [r.name for r in got] == [f"blocks.{i}" for i in range(4)]
    assert [r.tier for r in got] == ["gpu"] * 4
    assert [r.nbytes for r in got] == held


def test_a_block_is_already_freed_when_it_is_announced(tiny_checkpoint):
    _, path = tiny_checkpoint
    model, handle = _loaded(path, resident=ALL[:2], streamed=ALL[2:], dtype="bf16")
    handle.subscribe()
    pre = handle._prefetcher
    blocks = {f"blocks.{i}": block for i, block in enumerate(model.blocks)}
    freed_at_the_time = []
    original = pre.announce

    def announce(name, tier, nbytes):
        freed_at_the_time.append(all(p.is_meta for p in blocks[name].parameters()))
        original(name, tier, nbytes)

    pre.announce = announce
    handle.final_pass()
    model(torch.randn(3, 16).bfloat16())
    assert freed_at_the_time == [True] * 4


def test_the_profile_counts_the_freed_blocks_without_a_subscriber(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    monkeypatch.setenv("TRASH_PANDA_PROFILE", "1")
    model, handle = _loaded(path, streamed=ALL, dtype="bf16")
    handle.final_pass()
    model(torch.randn(3, 16).bfloat16())
    text = handle._prefetcher.report()
    assert "final pass freed 4 blocks" in text
    assert "ram 4 blocks" in text
    assert "queue" not in text.split("final pass freed")[1]


def test_the_profile_reports_blocks_that_were_never_wrapped(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    monkeypatch.setenv("TRASH_PANDA_PROFILE", "1")
    model, handle = _loaded(path, resident=ALL)                    # nothing is wrapped, so the prefetcher counts nothing
    handle.final_pass()
    model(torch.randn(3, 16))
    text = handle._prefetcher.report()
    assert "final pass freed 4 blocks" in text
    assert "gpu 4 blocks" in text


def test_the_profile_says_how_full_the_queue_got(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    monkeypatch.setenv("TRASH_PANDA_PROFILE", "1")
    model, handle = _loaded(path, streamed=ALL, dtype="bf16")
    q = handle.subscribe()
    handle.final_pass()
    model(torch.randn(3, 16).bfloat16())
    q.get_nowait()
    text = handle._prefetcher.report()
    assert "the queue held at most 4, 3 unread now" in text


def test_the_final_pass_still_frees_everything_with_a_subscriber(tiny_checkpoint):
    _, path = tiny_checkpoint
    model, handle = _loaded(path, resident=ALL[:1], streamed=ALL[1:3], disk=ALL[3:], dtype="bf16")
    q = handle.subscribe()
    handle.final_pass()
    model(torch.randn(3, 16).bfloat16())
    assert all(p.is_meta for p in model.blocks.parameters())
    assert len(_drain(q)) == 4
    handle.remove()
