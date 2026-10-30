"""Check the actual weight-transfer imports and GPU buffers before model load."""
import torch

# Explicit import surfaces errors hidden by checkpoint_engine/__init__.py.
from verl.checkpoint_engine.nccl_checkpoint_engine import NCCLCheckpointEngine
from verl.checkpoint_engine import CheckpointEngineRegistry
import cupy as cp

assert CheckpointEngineRegistry.get("nccl") is NCCLCheckpointEngine
assert torch.cuda.device_count() > 0, "No visible CUDA devices"
assert cp.cuda.runtime.getDeviceCount() == torch.cuda.device_count()
for device in range(torch.cuda.device_count()):
    with torch.cuda.device(device), cp.cuda.Device(device):
        tensor = torch.arange(16, dtype=torch.int32, device="cuda")
        buffer = cp.zeros(tensor.numel() * tensor.element_size(), dtype=cp.uint8)
        buffer[:] = cp.asarray(tensor.view(torch.uint8))
        torch.cuda.synchronize()
        assert (buffer.view(cp.int32).get() == tensor.cpu().numpy()).all()
print("NCCL", cp.cuda.nccl.get_version(), "PyTorch/CuPy buffer copy passed on",
      torch.cuda.device_count(), "GPUs; collective transfer still requires the training test")
