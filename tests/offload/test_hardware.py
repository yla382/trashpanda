import pytest

torch = pytest.importorskip("torch")

from trash_panda.offload.hardware import detect_hardware  # noqa: E402


def test_detect_hardware_reports_ram():
    hw = detect_hardware()
    assert hw.ram_total > 0
    assert 0 < hw.ram_available <= hw.ram_total
    assert "RAM" in hw.summary()
