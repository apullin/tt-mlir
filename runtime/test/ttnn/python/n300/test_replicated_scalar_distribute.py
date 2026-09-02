# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import ttrt
import ttrt.runtime
from ttrt.common.util import *

from .constants import FLATBUFFER_BASE_PATH
from ..utils import DeviceContext, Helper

MESH_SHAPE = [1, 2]


def create_multi_device_scalar(value, dtype):
    runtime_dtype = Binary.Program.to_data_type(dtype)
    shards = [
        torch.tensor(value, dtype=dtype) for _ in range(MESH_SHAPE[0] * MESH_SHAPE[1])
    ]

    runtime_tensor = ttrt.runtime.create_multi_device_host_tensor(
        [shard.data_ptr() for shard in shards],
        [],
        [],
        shards[0].element_size(),
        runtime_dtype,
        {},
        MESH_SHAPE,
    )
    return runtime_tensor, shards


def host_tensor_to_torch(runtime_tensor):
    shape = runtime_tensor.get_shape()
    dtype = ttrt_datatype_to_torch_dtype(runtime_tensor.get_dtype())
    torch_tensor = torch.empty(shape, dtype=dtype)
    ttrt.runtime.memcpy(torch_tensor.data_ptr(), runtime_tensor)
    return torch_tensor


def test_replicated_scalar_distribute_accepts_multi_device_input(
    helper: Helper, request
):
    assert ttrt.runtime.get_num_available_devices() == 2

    binary_path = os.path.join(
        FLATBUFFER_BASE_PATH, "replicated_scalar.mlir.tmp.ttnn"
    )
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    runtime_input, input_shards = create_multi_device_scalar(3.0, torch.float32)

    with DeviceContext(mesh_shape=MESH_SHAPE) as mesh_device:
        output = ttrt.runtime.submit(mesh_device, helper.binary.fbb, 0, [runtime_input])[
            0
        ]
        host_outputs = ttrt.runtime.to_host(output, untilize=True)

        assert len(host_outputs) == 1
        assert torch.allclose(
            host_tensor_to_torch(host_outputs[0]), torch.tensor(3.0)
        )

        ttrt.runtime.deallocate_tensor(output, force=True)

    # Keep borrowed shard buffers live until after runtime execution.
    assert len(input_shards) == 2
    helper.teardown()
