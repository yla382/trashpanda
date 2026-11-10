"""The Coordinator on a real GPU, with models whose blocks stay streamed, so the pinned pool is really used."""
import pytest

torch = pytest.importorskip("torch")

from .tiny_model import meta_model  # noqa: E402
from trash_panda.offload import Coordinator, Step  # noqa: E402
from trash_panda.offload.planner import Tier  # noqa: E402

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
demotes = pytest.mark.filterwarnings("ignore:trash-panda:RuntimeWarning")      # the huge headroom puts a block in RAM


@needs_gpu
@demotes
def test_every_step_that_streams_blocks_uses_the_buffers_the_first_one_pinned(tiny_checkpoint):
    ref, path = tiny_checkpoint
    # A headroom bigger than any GPU: when a preloaded model is entered, no block can be moved up to the GPU, so its
    # blocks stay streamed. (A model this small would otherwise fit on the GPU, and no block would use the pool.)
    options = dict(compute_dtype="fp32", resident_blocks=1, min_free_ram=0, headroom_vram="64GB")
    co = Coordinator(*(Step(name, meta_model, path, key=name, **options) for name in "abc"))   # own key: own activations
    x = torch.randn(3, 16, device="cuda")
    want = ref.cuda()(x)
    addresses = []
    with co:
        for name in "abc":
            with co.stage(name) as (model, p, _handle):
                assert torch.allclose(model(x), want, atol=1e-5)
                assert sum(b.tier != Tier.GPU for b in p.blocks) >= 3          # streamed, so the pool is in use
                addresses.append([b.data_ptr() if b is not None else None for b in co.pool.buffers])
    assert all(a is not None for a in addresses[0])                            # the first step pinned its buffers ...
    assert addresses[0] == addresses[1] == addresses[2]                        # ... and the others used those very ones
    assert co.pool is None


@needs_gpu
@demotes
def test_a_second_run_through_the_same_steps_keeps_using_the_same_buffers(tiny_checkpoint):
    ref, path = tiny_checkpoint
    options = dict(compute_dtype="fp32", resident_blocks=1, min_free_ram=0, headroom_vram="64GB")
    co = Coordinator(*(Step(name, meta_model, path, key=name, **options) for name in "ab"))
    x = torch.randn(3, 16, device="cuda")
    want = ref.cuda()(x)
    addresses = []
    with co:
        for name in ("a", "b", "a", "b"):                                     # two images
            with co.stage(name) as (model, _p, _handle):
                assert torch.allclose(model(x), want, atol=1e-5)
                addresses.append([b.data_ptr() if b is not None else None for b in co.pool.buffers])
    assert len({tuple(a) for a in addresses}) == 1
