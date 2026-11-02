"""Regression check using the installed vLLM loader and tiny CPU tensors."""
import torch
from vllm.model_executor.models.utils import AutoWeightsLoader
from verl.utils.vllm.streamed_weight_loading import validate_tied_weights_after_stream


def run(order, expect_error=False):
    model = torch.nn.Module()
    model.embedding = torch.nn.Linear(2, 2, bias=False)
    model.lm_head = torch.nn.Linear(2, 2, bias=False)
    model.lm_head.weight = model.embedding.weight
    target = torch.full((2, 2), 7.0)
    original = AutoWeightsLoader._check_skipped_aliases
    failed = False
    try:
        with validate_tied_weights_after_stream():
            for name in order:
                loader = AutoWeightsLoader(model)
                # Mirror the alias map produced for Qwen's tied vocabulary
                # embeddings, without allocating vocabulary-sized tensors.
                loader.aliased_params = {"lm_head.weight": "embedding.weight"}
                loader.load_weights([(name, target)])
    except ValueError as error:
        if "uninitialized" not in str(error):
            raise
        failed = True
    assert AutoWeightsLoader._check_skipped_aliases is original
    assert failed == expect_error, (order, failed, expect_error)
    if not failed:
        assert torch.equal(model.embedding.weight, target)
        assert model.lm_head.weight is model.embedding.weight


run(["embedding.weight", "lm_head.weight"])
run(["lm_head.weight", "embedding.weight"])
run(["lm_head.weight"], expect_error=True)
print("PASS: separate tied-weight buckets in either order; missing canonical weight still rejected")
