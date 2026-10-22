import pytest

torch = pytest.importorskip("torch")

from .tiny_model import meta_model  # noqa: E402
from trash_panda.offload.hardware import Hardware  # noqa: E402
from trash_panda.offload.planner import (  # noqa: E402
    Plan, Tier, _GB, _PINNED_BUFFERS, _auto_ram_reserve, parse_bytes, plan,
)


def hw(vram_free, ram_available):
    return Hardware(
        vram_total=vram_free, vram_free=vram_free, ram_total=ram_available, ram_available=ram_available
    )


def auto(path, hardware, **kw):
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

    small = plan(meta_model(), path, compute_dtype="fp32", hardware=hardware, activation_vram=1000, min_free_ram=0)
    big = plan(meta_model(), path, compute_dtype="fp32", hardware=hardware, activation_vram=1000 + block, min_free_ram=0)
    assert [b.tier for b in small.blocks] == [Tier.GPU, Tier.GPU, Tier.RAM, Tier.RAM]
    assert [b.tier for b in big.blocks] == [Tier.GPU, Tier.RAM, Tier.RAM, Tier.RAM]


def test_pinned_buffers_are_subtracted_from_the_ram_budget(tiny_checkpoint):
    _, path = tiny_checkpoint
    largest = max(b.stored_bytes for b in auto(path, hw(0, 10 ** 9)).blocks)
    hardware = machine(16 * _GB, 10 * _GB, vram_free=1)                # a GPU is present, with almost no room on it
    p = plan_with(path, hardware, resident_blocks=0, min_free_ram=0)   # every block is off the GPU
    assert p.ram_budget == 10 * _GB - _PINNED_BUFFERS * largest


def test_no_pinned_buffers_without_a_gpu(tiny_checkpoint):
    _, path = tiny_checkpoint
    p = plan_with(path, machine(16 * _GB, 10 * _GB, vram_free=0), min_free_ram=0)
    assert p.ram_budget == 10 * _GB                                    # no CUDA, so no pinned pool


def test_no_pinned_buffers_when_every_block_is_on_the_gpu(tiny_checkpoint):
    _, path = tiny_checkpoint
    p = plan_with(path, machine(16 * _GB, 10 * _GB, vram_free=1), resident_blocks=4, min_free_ram=0)
    assert all(b.tier == Tier.GPU for b in p.blocks)
    assert p.ram_budget == 10 * _GB                                    # nothing to stage, so nothing is pinned
