"""
tests/test_kernels.py — kernel test

run: pytest tests/test_kernels.py -v
"""
import pytest
import torch

cuda_available = torch.cuda.is_available()
try:
    import triton
    triton_available = True
except ImportError:
    triton_available = False

pytestmark = pytest.mark.skipif(
    not (cuda_available and triton_available), reason="CUDA + triton needed")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 7, 896), (2, 128, 896), (1, 1, 4096)])
def test_rmsnorm_triton_matches_reference(dtype, shape):
    from minivllm.kernels.rmsnorm import rmsnorm_triton
    from minivllm.models.layers import RMSNorm

    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cuda")
    norm = RMSNorm(shape[-1], eps=1e-6).to("cuda", dtype)
    with torch.no_grad():
        norm.weight.mul_(0.5).add_(0.75)

    ref = norm(x)
    out = rmsnorm_triton(x, norm.weight, 1e-6)

    tol = 1e-5 if dtype == torch.float32 else 2e-2
    assert torch.allclose(ref, out, atol=tol, rtol=tol), \
        f"max diff = {(ref - out).abs().max().item()}"


def test_decode_attention_kernel_placeholder():
    #TODO
    pytest.skip("after attention/triton_backend.py implementation")
