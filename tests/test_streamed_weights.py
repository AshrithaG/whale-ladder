"""Validate bucket ordering, module isolation, and restoration on failure."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "streamed_weight_loading",
    Path(__file__).resolve().parents[1] / "setup/streamed_weight_loading.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Loader:
    def __init__(self, model):
        self.module = model

    def _check_skipped_aliases(self, loaded_names):
        if "lm_head" in loaded_names and "embedding" not in loaded_names:
            raise ValueError("Missing embedding")


class StreamedWeights(unittest.TestCase):
    def test_either_bucket_order_still_requires_canonical_weight(self):
        for order in (("lm_head", "embedding"), ("embedding", "lm_head")):
            model = object()
            with module.validate_tied_weights_after_stream(Loader):
                for name in order:
                    Loader(model)._check_skipped_aliases({name})
        with self.assertRaisesRegex(ValueError, "Missing embedding"):
            with module.validate_tied_weights_after_stream(Loader):
                Loader(object())._check_skipped_aliases({"lm_head"})

    def test_other_module_cannot_supply_missing_weight(self):
        with self.assertRaisesRegex(ValueError, "Missing embedding"):
            with module.validate_tied_weights_after_stream(Loader):
                Loader(object())._check_skipped_aliases({"lm_head"})
                Loader(object())._check_skipped_aliases({"embedding"})

    def test_transfer_failure_restores_validation_and_preserves_error(self):
        original = Loader._check_skipped_aliases
        with self.assertRaisesRegex(RuntimeError, "transfer interrupted"):
            with module.validate_tied_weights_after_stream(Loader):
                Loader(object())._check_skipped_aliases({"lm_head"})
                raise RuntimeError("transfer interrupted")
        self.assertIs(Loader._check_skipped_aliases, original)
        with self.assertRaises(ValueError):
            Loader(object())._check_skipped_aliases({"lm_head"})

    def test_loader_without_alias_check_is_unchanged(self):
        class OlderLoader:
            pass
        with module.validate_tied_weights_after_stream(OlderLoader):
            self.assertFalse(hasattr(OlderLoader, "_check_skipped_aliases"))
        self.assertFalse(hasattr(OlderLoader, "_check_skipped_aliases"))


if __name__ == "__main__":
    unittest.main()
