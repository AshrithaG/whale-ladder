"""Preserve vLLM tied-weight validation across one bucketed weight update.

Installed into verl.utils.vllm by setup/install.sh. Only weight names and loader
references are retained; tensor buffers are still released between buckets.
"""
from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def validate_tied_weights_after_stream(loader_class=None):
    if loader_class is None:
        from vllm.model_executor.models.utils import AutoWeightsLoader
        loader_class = AutoWeightsLoader
    original = getattr(loader_class, "_check_skipped_aliases", None)
    if original is None:
        yield
        return
    groups = {}

    def collect(loader, loaded_names):
        # Nested loaders use names relative to their own module, so their
        # namespaces must not be mixed with the root model's names.
        names, checks = groups.setdefault(id(loader.module), (set(), []))
        names.update(loaded_names)
        checks.append(loader)

    with patch.object(loader_class, "_check_skipped_aliases", collect):
        yield
    # Runs only after a successful receive. The original validation is restored
    # even when transfer fails, and still rejects truly missing canonical names.
    for names, checks in groups.values():
        for loader in checks:
            original(loader, names)
