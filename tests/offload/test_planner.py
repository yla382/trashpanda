import pytest

torch = pytest.importorskip("torch")

from .tiny_model import meta_model  # noqa: E402
from trash_panda.offload.hardware import Hardware  # noqa: E402
from trash_panda.offload.planner import Plan, Tier, parse_bytes, plan  # noqa: E402


def hw(vram_free, ram_available):
    return Hardware(
        vram_total=vram_free, vram_free=vram_free, ram_total=ram_available, ram_available=ram_available
    )


def auto(path, hardware, **kw):
    return plan(meta_model(), path, compute_dtype="fp32", hardware=hardware,
                headroom_vram=0, min_free_ram=0, **kw)


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
