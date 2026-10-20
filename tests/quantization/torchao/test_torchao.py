# coding=utf-8
# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import subprocess
import sys
import textwrap

import pytest

from diffusers import TorchAoConfig

from ...testing_utils import (
    enable_full_determinism,
    is_quantization,
    is_torchao,
    is_torchao_available,
    require_torch,
    require_torch_accelerator,
    require_torchao_version_greater_or_equal,
)


if is_torchao_available():
    from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig


enable_full_determinism()


# Model-level TorchAO tests live in `tests/models/testing_utils/quantization.py` and
# pipeline-level ones in `tests/pipelines/testing_utils/quantization.py`. This module covers
# backend behavior that fits neither: config validation and custom device maps with cpu/disk offload.
@is_quantization
@is_torchao
@require_torch
@require_torch_accelerator
@require_torchao_version_greater_or_equal("0.15.0")
class TestTorchAoConfig:
    def test_to_dict(self):
        """
        Makes sure the config format is properly set
        """
        quantization_config = TorchAoConfig(Int4WeightOnlyConfig(version=2))
        torchao_orig_config = quantization_config.to_dict()
        assert "quant_type" in torchao_orig_config
        assert "quant_method" in torchao_orig_config

    def test_post_init_check(self):
        """
        Test that non-AOBaseConfig types are rejected
        """
        _ = TorchAoConfig(Int4WeightOnlyConfig())
        with pytest.raises(TypeError):
            _ = TorchAoConfig("int4_weight_only")

        with pytest.raises(TypeError):
            _ = TorchAoConfig(42)

    def test_repr(self):
        """
        Check that there is no error in the repr
        """
        quantization_config = TorchAoConfig(Int8WeightOnlyConfig(version=2), modules_to_not_convert=["conv"])
        quantization_repr = repr(quantization_config)
        assert "TorchAoConfig" in quantization_repr
        assert "torchao" in quantization_repr


@require_torch
class TestTorchAoQuantizerImportWithoutFqnToConfig:
    """
    Regression tests for https://github.com/huggingface/diffusers/issues/15085.

    Some torchao builds do not expose `FqnToConfig` from `torchao.quantization` (it is
    the renamed `ModuleFqnToConfig`, and not every release exposes both names). Importing
    diffusers must not fail in that case. Each test imports the TorchAO quantizer in a
    subprocess whose `torchao` package is a stub missing `FqnToConfig`.
    """

    def _run_with_stub_torchao(self, tmp_path, quantization_init, mode):
        stub_root = tmp_path / mode
        quantization_dir = stub_root / "torchao" / "quantization"
        quantization_dir.mkdir(parents=True)
        (stub_root / "torchao" / "__init__.py").write_text("")
        (quantization_dir / "__init__.py").write_text(quantization_init)
        dist_info_dir = stub_root / "torchao-0.15.0.dist-info"
        dist_info_dir.mkdir()
        (dist_info_dir / "METADATA").write_text("Metadata-Version: 2.1\nName: torchao\nVersion: 0.15.0\n")
        (dist_info_dir / "top_level.txt").write_text("torchao\n")

        script = textwrap.dedent(
            """
            import sys

            mode = sys.argv[1]

            import diffusers
            from diffusers.quantizers.torchao.torchao_quantizer import (
                FqnToConfig,
                _is_fqn_to_config,
                _resolve_fqn_to_config,
            )

            # The import path from the issue report must keep working.
            _ = diffusers.DDIMScheduler

            if mode == "missing":
                assert FqnToConfig is None, FqnToConfig
                assert not _is_fqn_to_config(object())
            else:
                import torchao.quantization as torchao_quantization

                assert FqnToConfig is torchao_quantization.ModuleFqnToConfig
                config = torchao_quantization.ModuleFqnToConfig({"_default": "fallback"})
                assert _is_fqn_to_config(config)
                assert _resolve_fqn_to_config(config, "some.module", "some.module.weight") == "fallback"

            print("ok")
            """
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(stub_root) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-c", script, mode], env=env, capture_output=True, text=True, timeout=600
        )
        assert result.returncode == 0, result.stderr

    def test_import_without_any_fqn_to_config(self, tmp_path):
        self._run_with_stub_torchao(
            tmp_path,
            "def quantize_(*args, **kwargs):\n    raise NotImplementedError\n",
            mode="missing",
        )

    def test_import_with_only_module_fqn_to_config(self, tmp_path):
        quantization_init = textwrap.dedent(
            """
            def quantize_(*args, **kwargs):
                raise NotImplementedError


            class ModuleFqnToConfig:
                def __init__(self, module_fqn_to_config=None):
                    self.module_fqn_to_config = module_fqn_to_config or {}
            """
        )
        self._run_with_stub_torchao(tmp_path, quantization_init, mode="renamed")
