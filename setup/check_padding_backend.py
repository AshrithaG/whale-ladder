"""Exercise the lazy training padding helpers, including CUDA gradients."""
import torch

from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input


def check(device):
    for mask_values in (
        [[0, 1, 1, 1], [1, 1, 0, 0]],
        [[1, 1, 1, 1], [1, 1, 1, 1]],
        [[0, 0, 0, 0], [0, 1, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0]],
    ):
        mask = torch.tensor(mask_values, device=device, dtype=torch.int64)
        x = torch.randn(2, 4, 3, device=device, requires_grad=True)
        values, indices, cumulative, maximum, *_ = unpad_input(x, mask)
        torch.testing.assert_close(values, x[mask.bool()])
        expected_cumulative = torch.cat((torch.zeros(1, device=device, dtype=torch.int32), mask.sum(1).cumsum(0).int()))
        torch.testing.assert_close(cumulative, expected_cumulative)
        assert maximum == mask.sum(1).max().item()
        torch.testing.assert_close(index_first_axis(rearrange(x, "b s d -> (b s) d"), indices), values)
        restored = pad_input(values, indices, 2, 4)
        torch.testing.assert_close(restored, x * mask[..., None])
        restored.square().sum().backward()
        torch.testing.assert_close(x.grad, 2 * x.detach() * mask[..., None])
    print(f"padding round-trip, indices, lengths and gradients pass on {device}")


if __name__ == "__main__":
    check("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the training padding preflight")
    check("cuda")
