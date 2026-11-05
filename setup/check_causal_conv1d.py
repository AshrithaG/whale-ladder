"""Require the compiled causal-conv1d extension and check CUDA training gradients."""
from importlib.metadata import version

import torch
import torch.nn.functional as F
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update  # noqa: F401
import causal_conv1d_cuda  # noqa: F401 -- fail on missing/incompatible binary


def check():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the causal-conv1d preflight')
    # Compare a small real forward/backward with PyTorch's reference operation.
    # Exercise the SiLU activation and BF16 used by Qwen's training path.
    torch.manual_seed(42)
    for dtype in (torch.float32, torch.bfloat16):
        inputs = [torch.randn(*shape, device='cuda', dtype=dtype, requires_grad=True)
                  for shape in ((2, 16, 67), (16, 4), (16,))]
        refs = [x.detach().clone().requires_grad_() for x in inputs]
        actual = causal_conv1d_fn(*inputs, activation='silu')
        x, weight, bias = refs
        expected = F.silu(F.conv1d(x, weight.unsqueeze(1), bias, padding=3, groups=16)[..., :67])
        tolerance = 5e-2 if dtype == torch.bfloat16 else 1e-3
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        dout = torch.randn_like(actual)
        grads = torch.autograd.grad(actual, inputs, dout)
        ref_grads = torch.autograd.grad(expected, refs, dout)
        for grad, ref in zip(grads, ref_grads):
            torch.testing.assert_close(grad, ref, atol=tolerance, rtol=tolerance)
    torch.cuda.synchronize()
    print(f'causal-conv1d {version("causal-conv1d")}: CUDA FP32/BF16 forward/backward passed')


if __name__ == '__main__':
    check()
