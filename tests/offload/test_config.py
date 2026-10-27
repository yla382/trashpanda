import dataclasses

import pytest

torch = pytest.importorskip("torch")

from .tiny_model import meta_model  # noqa: E402
from trash_panda.offload import config, hooks  # noqa: E402
from trash_panda.offload.blocks import find_blocks  # noqa: E402
from trash_panda.offload.hardware import Hardware  # noqa: E402
from trash_panda.offload.planner import Plan, _GB, _auto_ram_reserve, plan  # noqa: E402


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for f in dataclasses.fields(config.Defaults):
        monkeypatch.delenv(config.ENV_PREFIX + f.name.upper(), raising=False)


def test_the_shipped_defaults_are_the_values_the_library_used_before_the_config_existed():
    d = config.DEFAULTS
    assert (d.lookahead, d.pinned_buffers, d.blocks_in_flight) == (2, 2, 2)
    assert (d.headroom_vram, d.stream_buffer_factor) == ("1.5GB", 1.3)
    assert (d.activation_margin_fraction, d.activation_margin_min_gb) == (0.05, 0.5)
    assert (d.min_free_ram, d.ram_reserve_fraction) == ("auto", 0.08)
    assert (d.ram_reserve_min_gb, d.ram_reserve_max_gb, d.ram_reserve_unknown_gb) == (1.0, 4.0, 2.0)
    assert d.commit_warn_fraction == 0.9
    assert (d.find_blocks_min_fraction, d.find_blocks_min_blocks) == (0.01, 2)
    assert d.profile is False


def test_without_overrides_current_is_the_shipped_defaults():
    assert config.current() == config.DEFAULTS


def test_environment_values_are_read_as_the_type_of_the_setting(monkeypatch):
    monkeypatch.setenv("TRASH_PANDA_LOOKAHEAD", "0")
    monkeypatch.setenv("TRASH_PANDA_STREAM_BUFFER_FACTOR", "1.5")
    monkeypatch.setenv("TRASH_PANDA_HEADROOM_VRAM", " 3GB ")
    monkeypatch.setenv("TRASH_PANDA_PROFILE", "true")
    cfg = config.current()
    assert cfg.lookahead == 0 and isinstance(cfg.lookahead, int)
    assert cfg.stream_buffer_factor == 1.5
    assert cfg.headroom_vram == "3GB"
    assert cfg.profile is True


@pytest.mark.parametrize("text, expected", [("1", True), ("yes", True), ("ON", True), ("0", False), ("off", False)])
def test_boolean_settings_accept_the_usual_spellings(monkeypatch, text, expected):
    monkeypatch.setenv("TRASH_PANDA_PROFILE", text)
    assert config.current().profile is expected


def test_an_invalid_value_names_the_variable(monkeypatch):
    monkeypatch.setenv("TRASH_PANDA_LOOKAHEAD", "two")
    with pytest.raises(ValueError, match="TRASH_PANDA_LOOKAHEAD"):
        config.current()
    monkeypatch.setenv("TRASH_PANDA_LOOKAHEAD", "2")
    monkeypatch.setenv("TRASH_PANDA_PROFILE", "maybe")
    with pytest.raises(ValueError, match="TRASH_PANDA_PROFILE"):
        config.current()


def test_an_empty_value_is_ignored(monkeypatch):
    monkeypatch.setenv("TRASH_PANDA_LOOKAHEAD", "")
    assert config.current().lookahead == config.DEFAULTS.lookahead


def test_current_is_read_when_it_is_called_not_when_the_library_is_imported(monkeypatch):
    monkeypatch.setattr(config, "DEFAULTS", dataclasses.replace(config.DEFAULTS, lookahead=7))
    assert config.current().lookahead == 7
    assert Plan(blocks=[]).lookahead == 7


# -- the planner

def test_plan_takes_its_defaults_from_the_config(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    machine = Hardware(vram_total=10 ** 9, vram_free=10 ** 9, ram_total=10 ** 9, ram_available=10 ** 9)
    assert plan(meta_model(), path, compute_dtype="fp32", hardware=machine).lookahead == 2
    monkeypatch.setenv("TRASH_PANDA_LOOKAHEAD", "0")
    assert plan(meta_model(), path, compute_dtype="fp32", hardware=machine).lookahead == 0
    assert plan(meta_model(), path, compute_dtype="fp32", hardware=machine, lookahead=3).lookahead == 3   # an argument wins


def test_headroom_vram_from_the_config_is_used_unless_an_argument_is_given(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    machine = Hardware(vram_total=4 * _GB, vram_free=4 * _GB, ram_total=10 ** 9, ram_available=10 ** 9)
    base = plan(meta_model(), path, compute_dtype="fp32", hardware=machine)
    assert base.vram_budget == 4 * _GB - int(1.5e9)
    monkeypatch.setenv("TRASH_PANDA_HEADROOM_VRAM", "1GB")
    assert plan(meta_model(), path, compute_dtype="fp32", hardware=machine).vram_budget == 4 * _GB - int(1e9)
    assert plan(meta_model(), path, compute_dtype="fp32", hardware=machine, headroom_vram=0).vram_budget == 4 * _GB


def test_the_automatic_ram_reserve_follows_the_config(monkeypatch):
    big = Hardware(ram_total=128 * _GB, ram_available=64 * _GB)
    assert _auto_ram_reserve(big) == 4 * _GB
    monkeypatch.setenv("TRASH_PANDA_RAM_RESERVE_MAX_GB", "2")
    assert _auto_ram_reserve(big) == 2 * _GB
    monkeypatch.setenv("TRASH_PANDA_RAM_RESERVE_UNKNOWN_GB", "3")
    assert _auto_ram_reserve(Hardware(ram_total=0, ram_available=0)) == 3 * _GB


# -- the blocks and the hooks

def test_find_blocks_thresholds_follow_the_config(monkeypatch):
    assert find_blocks(meta_model())
    monkeypatch.setenv("TRASH_PANDA_FIND_BLOCKS_MIN_BLOCKS", "100")
    assert find_blocks(meta_model()) == []
    assert find_blocks(meta_model(), min_blocks=2)                 # an argument wins


def test_the_prefetcher_and_the_pool_follow_the_config(monkeypatch):
    monkeypatch.setenv("TRASH_PANDA_LOOKAHEAD", "5")
    monkeypatch.setenv("TRASH_PANDA_PINNED_BUFFERS", "4")
    assert hooks._Prefetcher(torch.device("cpu")).lookahead == 5
    assert hooks._Prefetcher(torch.device("cpu"), lookahead=1).lookahead == 1
    assert len(hooks._PinnedPool().buffers) == 4
    assert len(hooks._PinnedPool(3).buffers) == 3
