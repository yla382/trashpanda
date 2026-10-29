import pytest

torch = pytest.importorskip("torch")

from .tiny_model import meta_model  # noqa: E402
from trash_panda.offload.hardware import Hardware  # noqa: E402
from trash_panda.offload.planner import (  # noqa: E402
    BlockPlan, Plan, Tier, _GB, _PINNED_BUFFERS, _auto_ram_reserve, _pinned_size, _spread_disk_blocks, parse_bytes, plan,
)


def hw(vram_free, ram_available):
    return Hardware(
        vram_total=vram_free, vram_free=vram_free, ram_total=ram_available, ram_available=ram_available
    )


def auto(path, hardware, **kw):
    kw.setdefault("lookahead", 0)                         # most tests are about the block sizes alone
    return plan(meta_model(), path, compute_dtype="fp32", hardware=hardware,
                headroom_vram=0, min_free_ram=0, **kw)


def machine(ram_total, ram_available, vram_free=0):
    return Hardware(
        vram_total=vram_free, vram_free=vram_free, ram_total=ram_total, ram_available=ram_available
    )


def plan_with(path, hardware, **kw):
    """Like auto(), but leaves min_free_ram alone so the default ("auto") is what gets used."""
    return plan(meta_model(), path, compute_dtype="fp32", hardware=hardware, headroom_vram=0, **kw)


def test_parse_bytes():
    assert parse_bytes("9GB") == 9_000_000_000
    assert parse_bytes("8GiB") == 8 * 1024 ** 3
    assert parse_bytes("1.5 gb") == 1_500_000_000
    assert parse_bytes(123) == 123
    assert parse_bytes(None) is None


def test_everything_on_gpu_when_it_fits(tiny_checkpoint):
    _, path = tiny_checkpoint
    p = auto(path, hw(10 ** 9, 10 ** 9))
    assert all(b.tier == Tier.GPU for b in p.blocks)


def test_no_vram_means_ram(tiny_checkpoint):
    _, path = tiny_checkpoint
    p = auto(path, hw(0, 10 ** 9))
    assert all(b.tier == Tier.RAM for b in p.blocks)


def test_no_vram_and_no_ram_means_disk(tiny_checkpoint):
    _, path = tiny_checkpoint
    p = auto(path, hw(0, 0))
    assert all(b.tier == Tier.DISK for b in p.blocks)
    assert any("disk" in w for w in p.warnings)


def test_disk_can_be_forbidden(tiny_checkpoint):
    _, path = tiny_checkpoint
    p = auto(path, hw(0, 0), allow_disk=False)
    assert all(b.tier == Tier.RAM for b in p.blocks)
    assert any("disk is not allowed" in w for w in p.warnings)


def test_partial_gpu_residency_follows_the_budget(tiny_checkpoint):
    _, path = tiny_checkpoint
    probe = auto(path, hw(10 ** 9, 10 ** 9))
    block = probe.blocks[0].stored_bytes
    vram = probe.other_bytes + probe.stream_buffer_bytes + 2 * block + 8      # room for exactly two blocks
    p = auto(path, hw(vram, 10 ** 9))
    assert [b.tier for b in p.blocks] == [Tier.GPU, Tier.GPU, Tier.RAM, Tier.RAM]


def test_resident_blocks_override_wins(tiny_checkpoint):
    _, path = tiny_checkpoint
    p = auto(path, hw(0, 10 ** 9), resident_blocks=3)
    assert [b.tier for b in p.blocks] == [Tier.GPU, Tier.GPU, Tier.GPU, Tier.RAM]


def test_plans_can_be_edited(tiny_checkpoint):
    _, path = tiny_checkpoint
    p = auto(path, hw(0, 10 ** 9))
    p.assign(Tier.DISK, indices=[3])
    p.assign(Tier.GPU, prefixes=["blocks.0"])
    assert [b.tier for b in p.blocks] == [Tier.GPU, Tier.RAM, Tier.RAM, Tier.DISK]
    assert p.resident == ("blocks.0.",) and p.disk == ("blocks.3.",)


def test_manual_plan_normalizes_prefixes():
    p = Plan.manual(resident=["blocks.0"], streamed=["blocks.1."], disk=["blocks.2"], compute_dtype="fp32")
    assert p.source == "manual"
    assert p.resident == ("blocks.0.",)
    assert p.streamed == ("blocks.1.",)
    assert p.disk == ("blocks.2.",)
    assert p.compute_dtype is torch.float32


def test_summary_mentions_the_tiers(tiny_checkpoint):
    _, path = tiny_checkpoint
    text = auto(path, hw(0, 10 ** 9)).summary()
    assert "streamed from RAM" in text


def test_auto_ram_reserve_scales_with_the_machine():
    assert abs(_auto_ram_reserve(machine(16 * _GB, 8 * _GB)) - 0.08 * 16 * _GB) <= 1
    assert _auto_ram_reserve(machine(4 * _GB, 2 * _GB)) == 1 * _GB          # lower limit
    assert _auto_ram_reserve(machine(128 * _GB, 64 * _GB)) == 4 * _GB       # upper limit
    assert _auto_ram_reserve(machine(0, 0)) == 2 * _GB                      # unknown machine


def test_default_reserve_is_auto(tiny_checkpoint):
    _, path = tiny_checkpoint
    hardware = machine(16 * _GB, 10 * _GB)
    p = plan_with(path, hardware)
    assert p.ram_budget == 10 * _GB - _auto_ram_reserve(hardware)


def test_explicit_reserve_overrides_auto(tiny_checkpoint):
    _, path = tiny_checkpoint
    hardware = machine(16 * _GB, 10 * _GB)
    assert plan_with(path, hardware, min_free_ram=0).ram_budget == 10 * _GB
    assert plan_with(path, hardware, min_free_ram="0").ram_budget == 10 * _GB
    assert plan_with(path, hardware, min_free_ram="500MB").ram_budget == 10 * _GB - 500_000_000


def test_reserve_larger_than_available_ram_leaves_no_budget(tiny_checkpoint):
    _, path = tiny_checkpoint
    p = plan_with(path, machine(16 * _GB, _GB // 2))        # the reserve (about 1.3 GB) is more than is free
    assert p.ram_budget == 0
    assert all(b.tier == Tier.DISK for b in p.blocks)


def test_activation_vram_sets_the_headroom_on_a_small_card(tiny_checkpoint):
    _, path = tiny_checkpoint
    hardware = machine(16 * _GB, 8 * _GB, vram_free=4 * _GB)       # 5% of 4 GiB is below the 0.5 GiB minimum margin
    p = plan_with(path, hardware, activation_vram="2GB")
    assert p.vram_budget == 4 * _GB - (2_000_000_000 + _GB // 2)


def test_activation_vram_margin_grows_with_a_big_card(tiny_checkpoint):
    _, path = tiny_checkpoint
    hardware = machine(64 * _GB, 32 * _GB, vram_free=20 * _GB)     # 5% of 20 GiB is 1 GiB
    p = plan_with(path, hardware, activation_vram="2GB")
    assert abs(p.vram_budget - (20 * _GB - (2_000_000_000 + _GB))) <= 1


def test_without_activation_vram_the_headroom_is_unchanged(tiny_checkpoint):
    _, path = tiny_checkpoint
    hardware = machine(16 * _GB, 8 * _GB, vram_free=4 * _GB)
    assert plan_with(path, hardware).vram_budget == 4 * _GB        # plan_with sets headroom_vram=0
    p = plan(meta_model(), path, compute_dtype="fp32", hardware=hardware, headroom_vram="1GB")
    assert p.vram_budget == 4 * _GB - 1_000_000_000


def test_activation_vram_replaces_headroom_vram(tiny_checkpoint):
    _, path = tiny_checkpoint
    hardware = machine(16 * _GB, 8 * _GB, vram_free=4 * _GB)
    p = plan(meta_model(), path, compute_dtype="fp32", hardware=hardware,
             headroom_vram="3GB", activation_vram="1GB")
    assert p.vram_budget == 4 * _GB - (1_000_000_000 + _GB // 2)   # the 3GB headroom is ignored


def test_bigger_activations_leave_fewer_resident_blocks(tiny_checkpoint):
    _, path = tiny_checkpoint
    probe = auto(path, hw(10 ** 9, 10 ** 9))
    block = probe.blocks[0].stored_bytes
    base = probe.other_bytes + probe.stream_buffer_bytes
    vram_free = _GB // 2 + 1000 + base + 2 * block + 8              # room for two blocks when activations are 1000 bytes
    hardware = Hardware(vram_total=_GB, vram_free=vram_free, ram_total=10 ** 9, ram_available=10 ** 9)

    small = plan(meta_model(), path, compute_dtype="fp32", hardware=hardware, activation_vram=1000, min_free_ram=0, lookahead=0)
    big = plan(meta_model(), path, compute_dtype="fp32", hardware=hardware, activation_vram=1000 + block, min_free_ram=0, lookahead=0)
    assert [b.tier for b in small.blocks] == [Tier.GPU, Tier.GPU, Tier.RAM, Tier.RAM]
    assert [b.tier for b in big.blocks] == [Tier.GPU, Tier.RAM, Tier.RAM, Tier.RAM]


def test_pinned_buffers_are_subtracted_from_the_ram_budget(tiny_checkpoint):
    _, path = tiny_checkpoint
    largest = max(b.stored_bytes for b in auto(path, hw(0, 10 ** 9)).blocks)
    hardware = machine(16 * _GB, 10 * _GB, vram_free=1)                # a GPU is present, with almost no room on it
    p = plan_with(path, hardware, resident_blocks=0, min_free_ram=0)   # every block is off the GPU
    assert p.ram_budget == 10 * _GB - _PINNED_BUFFERS * _pinned_size(largest)


def test_pinned_size_rounds_up_to_the_next_power_of_two():
    assert _pinned_size(0) == 0
    assert _pinned_size(1) == 1
    assert _pinned_size(1000) == 1024
    assert _pinned_size(2 ** 30) == 2 ** 30
    assert _pinned_size(2 ** 30 + 1) == 2 ** 31


def test_pinned_buffers_are_counted_at_the_rounded_size(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    largest = max(b.stored_bytes for b in auto(path, hw(0, 10 ** 9)).blocks)
    monkeypatch.setattr("trash_panda.offload.planner._pinned_size", lambda n: n + 1000)   # a size that cannot be mistaken for the raw one
    hardware = machine(16 * _GB, 10 * _GB, vram_free=1)
    p = plan_with(path, hardware, resident_blocks=0, min_free_ram=0)
    assert p.ram_budget == 10 * _GB - _PINNED_BUFFERS * (largest + 1000)


def test_no_pinned_buffers_without_a_gpu(tiny_checkpoint):
    _, path = tiny_checkpoint
    p = plan_with(path, machine(16 * _GB, 10 * _GB, vram_free=0), min_free_ram=0)
    assert p.ram_budget == 10 * _GB                                    # no CUDA, so no pinned pool


def test_no_pinned_buffers_when_every_block_is_on_the_gpu(tiny_checkpoint):
    _, path = tiny_checkpoint
    p = plan_with(path, machine(16 * _GB, 10 * _GB, vram_free=1), resident_blocks=4, min_free_ram=0)
    assert all(b.tier == Tier.GPU for b in p.blocks)
    assert p.ram_budget == 10 * _GB                                    # nothing to stage, so nothing is pinned


def test_the_lookahead_reserves_vram_for_the_staged_blocks(tiny_checkpoint):
    _, path = tiny_checkpoint
    probe = auto(path, hw(10 ** 9, 10 ** 9))
    block = probe.blocks[0].stored_bytes
    vram = probe.other_bytes + probe.stream_buffer_bytes + 2 * block + 8      # room for exactly two resident blocks
    assert [b.tier for b in auto(path, hw(vram, 10 ** 9), lookahead=0).blocks] == [Tier.GPU, Tier.GPU, Tier.RAM, Tier.RAM]
    one = auto(path, hw(vram, 10 ** 9), lookahead=1)
    assert [b.tier for b in one.blocks] == [Tier.GPU, Tier.RAM, Tier.RAM, Tier.RAM]    # one block's room went to staging
    assert one.staging_bytes == block and one.lookahead == 1


def test_nothing_is_reserved_for_staging_when_everything_fits_on_the_gpu(tiny_checkpoint):
    _, path = tiny_checkpoint
    p = auto(path, hw(10 ** 9, 10 ** 9), lookahead=3)
    assert all(b.tier == Tier.GPU for b in p.blocks)
    assert p.staging_bytes == 0


def test_the_summary_mentions_the_room_for_staged_blocks(tiny_checkpoint):
    _, path = tiny_checkpoint
    assert "staged ahead (look-ahead 2)" in auto(path, hw(0, 10 ** 9), lookahead=2).summary()
    assert "staged ahead" not in auto(path, hw(0, 10 ** 9), lookahead=0).summary()


def _tiers(p):
    return [b.tier for b in p.blocks]


def test_disk_blocks_are_spread_out_by_default(tiny_checkpoint):
    _, path = tiny_checkpoint
    block = auto(path, hw(10 ** 9, 10 ** 9)).blocks[0].stored_bytes
    p = auto(path, hw(0, 2 * block + 8))                       # room in RAM for two of the four blocks
    assert _tiers(p) == [Tier.RAM, Tier.DISK, Tier.RAM, Tier.DISK]


def test_disk_blocks_can_stay_at_the_end(tiny_checkpoint):
    _, path = tiny_checkpoint
    block = auto(path, hw(10 ** 9, 10 ** 9)).blocks[0].stored_bytes
    p = auto(path, hw(0, 2 * block + 8), spread_disk=False)
    assert _tiers(p) == [Tier.RAM, Tier.RAM, Tier.DISK, Tier.DISK]


def test_spreading_uses_no_disk_when_everything_fits_and_all_of_it_when_nothing_does(tiny_checkpoint):
    _, path = tiny_checkpoint
    assert all(t == Tier.RAM for t in _tiers(auto(path, hw(0, 10 ** 9))))
    assert all(t == Tier.DISK for t in _tiers(auto(path, hw(0, 0))))


def test_disk_is_not_used_when_it_is_forbidden_even_with_spreading(tiny_checkpoint):
    _, path = tiny_checkpoint
    p = auto(path, hw(0, 0), allow_disk=False)
    assert all(t == Tier.RAM for t in _tiers(p))


def test_the_helper_picks_evenly_spaced_blocks_and_as_few_as_needed():
    rest = [BlockPlan(f"b{i}.", stored_bytes=4) for i in range(6)]
    assert _spread_disk_blocks(rest, 24) == set()              # everything fits
    assert _spread_disk_blocks(rest, 20) == {3}                # one block has to go
    assert _spread_disk_blocks(rest, 12) == {1, 3, 5}          # three have to go, and they are evenly spaced
    assert _spread_disk_blocks(rest, 0) == set(range(6))       # nothing fits


def test_the_helper_counts_the_real_sizes_of_the_blocks():
    sizes = [1, 1, 10, 1, 1, 10]
    rest = [BlockPlan(f"b{i}.", stored_bytes=s) for i, s in enumerate(sizes)]
    chosen = _spread_disk_blocks(rest, 5)
    assert sum(s for i, s in enumerate(sizes) if i not in chosen) <= 5     # what stays in RAM fits the budget


def _gpu_warnings(p):
    return [w for w in p.warnings if "GPU part of the plan" in w]


def test_a_plan_that_leaves_plenty_of_vram_free_does_not_warn(tiny_checkpoint):
    _, path = tiny_checkpoint
    p = auto(path, hw(10 ** 9, 10 ** 9))                      # the tiny model uses a sliver of the card
    assert _gpu_warnings(p) == []


def test_a_plan_that_fills_the_free_vram_warns(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    monkeypatch.setenv("TRASH_PANDA_VRAM_FILL_WARN_FRACTION", "0.0")      # any use of the card counts as filling it
    p = auto(path, hw(10 ** 9, 10 ** 9))
    found = _gpu_warnings(p)
    assert len(found) == 1 and "fills" in found[0] and "little room for activations" in found[0]


def test_the_fill_warning_follows_the_configured_share(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    monkeypatch.setenv("TRASH_PANDA_VRAM_FILL_WARN_FRACTION", "2.0")      # never reached
    assert _gpu_warnings(auto(path, hw(10 ** 9, 10 ** 9))) == []


def test_a_plan_over_the_budget_gets_one_warning_not_two(tiny_checkpoint):
    _, path = tiny_checkpoint
    p = auto(path, hw(1000, 10 ** 9), resident_blocks=4)                  # forced onto a card that cannot hold them
    found = _gpu_warnings(p)
    assert len(found) == 1 and "needs" in found[0] and "budget" in found[0]
