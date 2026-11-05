"""Optional GPU diagnostic: real solver and actor forward/backward on 64 tokens.

Run with the pinned local model snapshot as the first argument. Add
--native-offload to exercise FSDP2 accumulation and a CPU AdamW step as well.
This does not replace testing real workload memory, Ray, and checkpointing.
"""
import sys
import tempfile
import time

import torch
from omegaconf import OmegaConf
from transformers import AutoModelForImageTextToText

from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.workers.actor.dp_actor import DataParallelPPOActor


def main():
    started = time.monotonic()
    native_offload = "--native-offload" in sys.argv[2:]
    with tempfile.TemporaryDirectory() as tmp:
        torch.distributed.init_process_group("cpu:gloo,cuda:nccl", init_method=f"file://{tmp}/dist", rank=0, world_size=1)
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                sys.argv[1], dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
            ).cuda().train()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            apply_monkey_patch(model, use_remove_padding=True, ulysses_sp_size=1, use_fused_kernels=False)
            if native_offload:
                from torch.distributed.device_mesh import init_device_mesh
                from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
                from verl.utils.fsdp_utils import apply_fsdp2, fsdp2_load_full_state_dict, get_shard_placement_fn
                mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("fsdp",))
                policy = CPUOffloadPolicy(pin_memory=True)
                full_state = model.state_dict()
                apply_fsdp2(model, {
                    "mesh": mesh,
                    "mp_policy": MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
                    "offload_policy": policy, "reshard_after_forward": False,
                    "shard_placement_fn": get_shard_placement_fn(1),
                }, OmegaConf.create({"wrap_policy": {}}))
                fsdp2_load_full_state_dict(model, full_state, mesh, policy)
                del full_state
                torch.cuda.empty_cache()
            config = OmegaConf.create({
                "use_remove_padding": True, "use_fused_kernels": False,
                "ulysses_sequence_parallel_size": 1, "use_dynamic_bsz": False,
                "entropy_from_logits_with_chunking": False, "use_torch_compile": False,
                "fsdp_config": {"dtype": "bfloat16"},
                "grad_clip": 1.0,
            })
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, foreach=False) if native_offload else None
            actor = DataParallelPPOActor(config, model, optimizer)
            ids = torch.arange(100, 164, device="cuda").unsqueeze(0)
            mask = torch.ones_like(ids)
            mask[:, :4] = 0
            positions = (mask.cumsum(-1) - 1).clamp_min(0)
            batch = {"input_ids": ids, "attention_mask": mask, "position_ids": positions,
                     "responses": ids[:, -16:]}
            output = actor._forward_micro_batch(batch, temperature=1.0)
            loss = -output["log_probs"].mean()
            assert torch.isfinite(loss).item(), loss
            loss.backward()
            grads = [p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
                     for p in model.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all().item() for g in grads)
            assert any(torch.count_nonzero(g).item() for g in grads)
            if native_offload:
                # Exercise accumulation as used by the micro-batch training loop.
                probe = next(p for p in model.parameters() if p.grad is not None)
                initial_grad = probe.grad.to_local().flatten()[:1024].clone()
                second = -actor._forward_micro_batch(batch, temperature=1.0)["log_probs"].mean()
                second.backward()
                torch.testing.assert_close(probe.grad.to_local().flatten()[:1024], 2 * initial_grad)
                before = probe.detach().to_local().flatten()[:1024].clone()
                norm = actor._optimizer_step()
                assert torch.isfinite(norm).item()
                assert all(p.device.type == "cpu" for p in model.parameters())
                assert optimizer.state
                assert all(value.device.type == "cpu" for state in optimizer.state.values()
                           for value in state.values() if torch.is_tensor(value))
                after = probe.detach().to_local().flatten()[:1024]
                assert not torch.equal(before, after), "optimizer did not change probe weights"
                print("PASS native CPU optimizer state, accumulated backward and AdamW update", flush=True)
            torch.cuda.synchronize()
            print(f"PASS real 4B actor forward/backward: loss={loss.item():.4f}, "
                  f"seconds={time.monotonic()-started:.1f}, "
                  f"peak_allocated_GiB={torch.cuda.max_memory_allocated()/2**30:.2f}", flush=True)
        finally:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
