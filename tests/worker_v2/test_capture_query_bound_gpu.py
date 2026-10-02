# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend, FlashAttentionImpl
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.utils import AttentionGroup

from tests.helpers.mark import hardware_test
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

pytestmark = [pytest.mark.core_model]


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize(
    "graph_mode", [CUDAGraphMode.FULL, CUDAGraphMode.FULL_AND_PIECEWISE, CUDAGraphMode.FULL_DECODE_ONLY]
)
def test_graph_query_bound_uses_real_flash_attention_backend(graph_mode):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("requires FA3 on Hopper or newer")
    _check_replay(graph_mode)


def _check_replay(graph_mode):
    torch.manual_seed(17)
    device = torch.device("cuda:0")
    decode = graph_mode.separate_routine()
    num_tokens = 64 if decode else 256
    runtime_mode = CUDAGraphMode.FULL if decode else CUDAGraphMode.NONE
    c = VllmConfig()
    c.model_config = SimpleNamespace(
        get_num_attention_heads=lambda p: 32,
        rswa_window=None,
        is_diffusion=False,
        is_mm_prefix_lm=False,
        hf_config=SimpleNamespace(model_type="qwen3"),
    )
    c.scheduler_config.max_num_seqs = 64
    c.compilation_config.cudagraph_mode = graph_mode
    c.compilation_config.max_cudagraph_capture_size = 256
    c.attention_config.flash_attn_version = 3
    spec = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=128, dtype=torch.bfloat16)
    kvconfig = KVCacheConfig(num_blocks=1024, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(["attn"], spec)])
    state = OmniModelState.__new__(OmniModelState)
    state.vllm_config = c
    state.max_model_len = 256
    state.supports_mm_inputs = False
    with set_current_vllm_config(c):
        group = AttentionGroup(FlashAttentionBackend, ["attn"], spec, 0)
        group.create_metadata_builders(c, device)
        impl = FlashAttentionImpl(32, 128, 128**-0.5, 8, None, None, "auto")
        assert impl.vllm_flash_attn_version == 3
        buffers = InputBuffers(64, 256, device)
        batch = InputBatch.make_dummy(64, num_tokens, buffers)
        tables = torch.arange(1024, dtype=torch.int32, device=device).reshape(64, 16)
        slots = torch.zeros((1, num_tokens), dtype=torch.int64, device=device)
        cache = torch.randn(1024, 8, 16, 256, device=device, dtype=torch.bfloat16)
        q = torch.randn(num_tokens, 32, 128, device=device, dtype=torch.bfloat16)
        k = v = torch.empty(num_tokens, 8, 128, device=device, dtype=torch.bfloat16)
        layer = SimpleNamespace(
            _q_scale=torch.ones((), device=device),
            _k_scale=torch.ones((), device=device),
            _v_scale=torch.ones((), device=device),
        )
        prepare = OmniModelState.prepare_attn
        md = prepare(state, batch, runtime_mode, (tables,), slots, [[group]], kvconfig, for_capture=True)["attn"]
        out = torch.full_like(q, float("nan"))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                impl.forward(layer, q, k, v, cache, md, out)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            impl.forward(layer, q, k, v, cache, md, out)
        # Reuse the captured shape for long prefill with padding, or Q=1 decode
        # with longer KV contexts. No attention backend is mocked.
        lengths = (
            np.arange(37, 101, dtype=np.int32) if decode else np.array([212] + [1] * 44 + [0] * 19, dtype=np.int32)
        )
        queries = np.ones(64, dtype=np.int32) if decode else lengths
        query_start_loc = np.r_[0, np.cumsum(queries)]
        batch.query_start_loc.copy_(torch.tensor(query_start_loc, device=device, dtype=torch.int32))
        batch.seq_lens.copy_(torch.tensor(lengths, device=device))
        from dataclasses import replace

        runtime = replace(
            batch,
            max_query_len=int(queries.max()),
            num_scheduled_tokens=queries,
            query_start_loc_np=query_start_loc,
            seq_lens_cpu_upper_bound=torch.tensor(lengths),
        )
        eager_md = state.prepare_attn(runtime, runtime_mode, (tables,), slots, [[group]], kvconfig)["attn"]
        expected = torch.empty_like(q)
        impl.forward(layer, q, k, v, cache, eager_md, expected)
        state.prepare_attn(runtime, runtime_mode, (tables,), slots, [[group]], kvconfig)
        out.fill_(float("nan"))
        graph.replay()
        torch.accelerator.synchronize()
        assert md.max_query_len == (1 if decode else 256)
        valid_tokens = 64 if decode else 212
        torch.testing.assert_close(out[:valid_tokens], expected[:valid_tokens], atol=0.002, rtol=0.002)
