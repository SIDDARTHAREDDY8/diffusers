# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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
import logging
import tempfile
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file
from transformers import CLIPTextConfig, CLIPTextModel, T5Config, T5EncoderModel

from diffusers import ChromaPipeline, ChromaTransformer2DModel
from diffusers.configuration_utils import ConfigMixin
from diffusers.loaders import StableDiffusionLoraLoaderMixin, lora_base
from diffusers.loaders.lora_base import LoraBaseMixin
from diffusers.loaders.peft import PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import is_transformers_version
from diffusers.utils.import_utils import is_peft_available
from diffusers.utils.state_dict_utils import convert_state_dict_to_diffusers, convert_state_dict_to_peft

from ..testing_utils import CaptureLogger, require_peft_backend


if is_peft_available():
    from peft import LoraConfig
    from peft.tuners.tuners_utils import BaseTunerLayer
    from peft.utils import get_peft_model_state_dict


LORA_KEY = "unet.test.lora_A.weight"


def _write_lora_weights(path):
    save_file({LORA_KEY: torch.ones(1)}, path)


@pytest.fixture
def lora_weight_path(tmp_path):
    weight_path = tmp_path / "adapter.safetensors"
    _write_lora_weights(weight_path)
    return weight_path


@pytest.fixture
def model_info_mock(monkeypatch):
    model_info_mock = Mock()
    monkeypatch.setattr(lora_base, "model_info", model_info_mock)
    return model_info_mock


def test_local_directory_in_offline_mode(lora_weight_path, monkeypatch, model_info_mock):
    monkeypatch.setattr(lora_base, "HF_HUB_OFFLINE", True)

    state_dict, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(lora_weight_path.parent)

    assert torch.equal(state_dict[LORA_KEY], torch.ones(1))
    model_info_mock.assert_not_called()


def test_local_directory_with_local_files_only(lora_weight_path, model_info_mock):
    state_dict, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(lora_weight_path.parent, local_files_only=True)

    assert torch.equal(state_dict[LORA_KEY], torch.ones(1))
    model_info_mock.assert_not_called()


def test_local_file_in_offline_mode(lora_weight_path, monkeypatch, model_info_mock):
    monkeypatch.setattr(lora_base, "HF_HUB_OFFLINE", True)

    state_dict, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(lora_weight_path)

    assert torch.equal(state_dict[LORA_KEY], torch.ones(1))
    model_info_mock.assert_not_called()


def test_remote_repository_in_offline_mode_requires_weight_name(monkeypatch, model_info_mock):
    monkeypatch.setattr(lora_base, "HF_HUB_OFFLINE", True)

    with pytest.raises(ValueError, match="offline mode.*weight_name"):
        StableDiffusionLoraLoaderMixin.lora_state_dict("organization/repository")

    model_info_mock.assert_not_called()


def test_local_directory_without_matching_files_returns_none(tmp_path, monkeypatch):
    (tmp_path / "notes.txt").touch()
    monkeypatch.setattr(lora_base, "HF_HUB_OFFLINE", True)

    weight_name = lora_base._best_guess_weight_name(tmp_path)

    assert weight_name is None


def test_local_directory_with_multiple_files_warns_and_uses_first(tmp_path, monkeypatch):
    first_path = tmp_path / "first.safetensors"
    second_path = tmp_path / "second.safetensors"
    first_path.touch()
    second_path.touch()
    monkeypatch.setattr(lora_base, "HF_HUB_OFFLINE", True)
    # `os.listdir` returns entries in arbitrary order; pin it so `first.safetensors` is the one picked.
    monkeypatch.setattr(lora_base.os, "listdir", lambda _: [first_path.name, second_path.name])

    # `caplog` cannot see this warning: `diffusers` sets `propagate = False` on its library root logger, so
    # records never reach the root handler pytest installs. `CaptureLogger` attaches to the logger directly.
    lora_base.logger.setLevel(logging.WARNING)
    with CaptureLogger(lora_base.logger) as cap_logger:
        weight_name = lora_base._best_guess_weight_name(tmp_path)

    assert weight_name == first_path.name
    assert "contains more than one weights file" in cap_logger.out


@require_peft_backend
def test_unfuse_lora_partial_components_keeps_merged_adapters_in_sync():
    """Regression test for gh-14214.

    Unfusing only a subset of components must keep _merged_adapters in sync
    with the adapters still physically fused in the remaining components.
    """

    class TinyModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
        config_name = "config.json"

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8)

    class FakePipeline(LoraBaseMixin):
        _lora_loadable_modules = ["unet", "text_encoder"]

        def __init__(self, unet, text_encoder):
            self._merged_adapters = set()
            self.unet, self.text_encoder = unet, text_encoder

    unet = TinyModel()
    text_encoder = TinyModel()
    config = LoraConfig(r=4, lora_alpha=4, target_modules=["linear"], init_lora_weights=False)
    unet.add_adapter(config, adapter_name="adapter")
    text_encoder.add_adapter(config, adapter_name="adapter")

    pipe = FakePipeline(unet, text_encoder)
    pipe.fuse_lora(components=["unet", "text_encoder"], adapter_names=["adapter"])
    assert pipe.num_fused_loras == 1

    pipe.unfuse_lora(components=["text_encoder"])
    assert "adapter" in pipe.fused_loras, "adapter should remain tracked while unet is still fused"
    assert pipe.num_fused_loras == 1

    unet_still_merged = any(isinstance(m, BaseTunerLayer) and len(m.merged_adapters) > 0 for m in unet.modules())
    assert unet_still_merged, "unet should still be physically merged at the PEFT level"

    pipe.unfuse_lora(components=["unet"])
    assert pipe.num_fused_loras == 0


@require_peft_backend
@pytest.mark.skipif(
    not is_transformers_version(">=", "5.6.0"),
    reason="CLIPTextModel uses a flattened module namespace starting with transformers 5.6",
)
def test_load_lora_into_flattened_clip_text_encoder():
    config = CLIPTextConfig(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=77,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
    )
    text_encoder = CLIPTextModel(config)

    lora_a = torch.randn(4, 32)
    lora_b = torch.randn(32, 4)

    state_dict = {
        "text_encoder.text_model.encoder.layers.0.self_attn.q_proj.lora_A.weight": lora_a,
        "text_encoder.text_model.encoder.layers.0.self_attn.q_proj.lora_B.weight": lora_b,
    }

    StableDiffusionLoraLoaderMixin.load_lora_into_text_encoder(
        state_dict=state_dict,
        network_alphas=None,
        text_encoder=text_encoder,
        prefix="text_encoder",
        adapter_name="test",
        low_cpu_mem_usage=False,
    )

    q_proj = text_encoder.encoder.layers[0].self_attn.q_proj

    torch.testing.assert_close(q_proj.lora_A["test"].weight, lora_a)
    torch.testing.assert_close(q_proj.lora_B["test"].weight, lora_b)


def test_convert_state_dict_with_non_clip_module_names():
    # PEFT state dicts of text encoders whose modules are not CLIP-named (T5, SmolLM3, ChatGLM)
    # are already in PEFT format and must not fail with "Could not automatically infer state dict
    # type", regardless of the module names they contain.
    weight = torch.randn(4, 4)
    for state_dict in [
        # T5 (Chroma, DeepFloyd IF, Kandinsky 3)
        {"encoder.block.0.layer.0.SelfAttention.q.lora_A.weight": weight},
        {"encoder.block.0.layer.0.SelfAttention.o.lora_B.weight": weight},
        {"encoder.block.0.layer.1.DenseReluDense.wi_0.lora_A.weight": weight},
        # SmolLM3 (Bria FIBO), including an adapter that only targets `o_proj`
        {"model.layers.0.self_attn.o_proj.lora_A.weight": weight},
        # ChatGLM (Kolors)
        {"transformer.encoder.layers.0.attention.query_key_value.lora_B.weight": weight},
    ]:
        assert convert_state_dict_to_peft(dict(state_dict)) == state_dict
        # The PEFT -> diffusers -> PEFT round-trip restores the original keys.
        diffusers_state_dict = convert_state_dict_to_diffusers(dict(state_dict))
        assert convert_state_dict_to_peft(diffusers_state_dict) == state_dict


@require_peft_backend
def test_load_lora_into_t5_text_encoder_with_mixed_ranks():
    # Regression test for gh-15020: a T5 text-encoder LoRA whose modules have different ranks
    # must load with each module's own rank, not the most frequent one.
    text_encoder = T5EncoderModel(
        T5Config(vocab_size=32, d_model=16, d_kv=4, d_ff=32, num_layers=1, num_heads=2)
    )

    # `q` has rank 4, `o` has rank 8.
    state_dict = {
        "text_encoder.encoder.block.0.layer.0.SelfAttention.q.lora_A.weight": torch.randn(4, 16),
        "text_encoder.encoder.block.0.layer.0.SelfAttention.q.lora_B.weight": torch.randn(8, 4),
        "text_encoder.encoder.block.0.layer.0.SelfAttention.o.lora_A.weight": torch.randn(8, 8),
        "text_encoder.encoder.block.0.layer.0.SelfAttention.o.lora_B.weight": torch.randn(16, 8),
    }

    StableDiffusionLoraLoaderMixin.load_lora_into_text_encoder(
        state_dict=state_dict,
        network_alphas=None,
        text_encoder=text_encoder,
        prefix="text_encoder",
        adapter_name="test",
        low_cpu_mem_usage=False,
    )

    self_attention = text_encoder.encoder.block[0].layer[0].SelfAttention
    for module_name, rank in [("q", 4), ("o", 8)]:
        module = getattr(self_attention, module_name)
        expected_a = state_dict[f"text_encoder.encoder.block.0.layer.0.SelfAttention.{module_name}.lora_A.weight"]
        expected_b = state_dict[f"text_encoder.encoder.block.0.layer.0.SelfAttention.{module_name}.lora_B.weight"]
        assert module.lora_A["test"].weight.shape[0] == rank
        torch.testing.assert_close(module.lora_A["test"].weight, expected_a)
        torch.testing.assert_close(module.lora_B["test"].weight, expected_b)


@require_peft_backend
def test_t5_text_encoder_lora_save_load_round_trip_with_mixed_ranks():
    # Regression test for gh-15020: diffusers could not load back a T5 text-encoder LoRA it had
    # saved itself with `save_lora_weights`.

    def tiny_pipe():
        text_encoder = T5EncoderModel(
            T5Config(vocab_size=32, d_model=16, d_kv=4, d_ff=32, num_layers=2, num_heads=2)
        )
        transformer = ChromaTransformer2DModel(
            patch_size=1,
            in_channels=4,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=16,
            axes_dims_rope=[4, 4, 8],
            approximator_hidden_dim=32,
            approximator_layers=1,
            approximator_num_channels=16,
        )
        return ChromaPipeline(
            scheduler=None, text_encoder=text_encoder, tokenizer=None, transformer=transformer, vae=None
        )

    pipe = tiny_pipe()
    pipe.text_encoder.add_adapter(
        LoraConfig(
            r=4,
            lora_alpha=4,
            target_modules=["q", "k", "v", "o"],
            rank_pattern={"o": 8},
            alpha_pattern={"o": 8},
        )
    )
    lora_layers = get_peft_model_state_dict(pipe.text_encoder)
    # Sanity check that the adapter really has mixed ranks.
    assert {v.shape[1] for k, v in lora_layers.items() if "lora_B" in k} == {4, 8}

    with tempfile.TemporaryDirectory() as tmpdir:
        ChromaPipeline.save_lora_weights(tmpdir, text_encoder_lora_layers=lora_layers)
        loaded_pipe = tiny_pipe()
        loaded_pipe.load_lora_weights(tmpdir)

    adapter_name = next(iter(loaded_pipe.text_encoder.peft_config))
    loaded_lora_layers = get_peft_model_state_dict(loaded_pipe.text_encoder, adapter_name=adapter_name)
    assert loaded_lora_layers.keys() == lora_layers.keys()
    for key, weight in lora_layers.items():
        torch.testing.assert_close(loaded_lora_layers[key], weight)
