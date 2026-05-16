"""Tests for braggtrack.io.paths (dataset root resolution)."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from braggtrack.io.paths import default_dataset_root, resolve_dataset_root, sample_operando_root


class SampleOperandoRootTests(unittest.TestCase):
    def test_env_var_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"BRAGGTRACK_DATA_ROOT": tmpdir}):
                result = sample_operando_root()
                self.assertEqual(result, Path(tmpdir))

    def test_env_var_nonexistent_falls_back(self) -> None:
        with patch.dict(os.environ, {"BRAGGTRACK_DATA_ROOT": "/nonexistent/path/xyz"}):
            result = sample_operando_root()
            self.assertNotEqual(result, Path("/nonexistent/path/xyz"))

    def test_no_env_var_returns_default(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BRAGGTRACK_DATA_ROOT", None)
            result = sample_operando_root()
            self.assertTrue(str(result).endswith("sample_operando"))


class ResolveDatasetRootTests(unittest.TestCase):
    def test_explicit_path_wins(self) -> None:
        result = resolve_dataset_root("/tmp/my_data")
        self.assertEqual(result, Path("/tmp/my_data"))

    def test_none_delegates_to_default(self) -> None:
        result = resolve_dataset_root(None)
        self.assertIsInstance(result, Path)

    def test_default_returns_path(self) -> None:
        result = default_dataset_root()
        self.assertIsInstance(result, Path)


if __name__ == "__main__":
    unittest.main()
