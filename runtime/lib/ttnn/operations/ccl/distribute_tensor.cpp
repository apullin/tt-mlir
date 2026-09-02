// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/distributed/distributed_tensor.hpp"

#include <algorithm>
#include <variant>

namespace tt::runtime::ttnn::operations::ccl {
using ::ttnn::distributed::MeshMapperConfig;
using ::ttnn::distributed::TensorToMesh;

namespace {
bool isFullyReplicated(const MeshMapperConfig &config) {
  return !config.placements.empty() &&
         std::all_of(config.placements.begin(), config.placements.end(),
                     [](const MeshMapperConfig::Placement &placement) {
                       return std::holds_alternative<
                           MeshMapperConfig::Replicate>(placement);
                     });
}

bool isAlreadyFullyReplicatedOnMesh(const ::ttnn::Tensor &input,
                                    const ::ttnn::MeshDevice &meshDevice,
                                    const MeshMapperConfig &config) {
  if (!isFullyReplicated(config) || !input.is_scalar()) {
    return false;
  }

  const auto &hostTensor = input.host_tensor();
  return hostTensor.buffer().shape() == meshDevice.shape();
}
} // namespace

void run(const ::tt::target::ttnn::DistributeTensorOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());

  LOG_ASSERT(
      ttnn::utils::isOnHost(input.storage_type()),
      "Input of distribute_tensor must be HOST. id:", op->in()->global_id());
  ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();

  std::optional<::ttnn::QueueId> cqId = std::nullopt;
  if (op->cq_id().has_value()) {
    cqId = ::ttnn::QueueId(op->cq_id().value());
  }

  MeshMapperConfig meshMapperConfig;

  auto *placements = op->mapper_config()->placements();
  for (const auto *placement : *placements) {
    if (placement->type() == ::tt::target::ttnn::PlacementType::Replicate) {
      meshMapperConfig.placements.push_back(MeshMapperConfig::Replicate());
    } else if (placement->type() == ::tt::target::ttnn::PlacementType::Shard) {
      meshMapperConfig.placements.push_back(
          MeshMapperConfig::Shard{placement->dim()});
    }
  }

  if (op->mapper_config()->mesh_shape_override()) {
    ::ttsl::SmallVector<uint32_t> meshShapeOverride(
        op->mapper_config()->mesh_shape_override()->begin(),
        op->mapper_config()->mesh_shape_override()->end());
    meshMapperConfig.mesh_shape_override = ::ttnn::MeshShape(meshShapeOverride);
  }
  std::unique_ptr<TensorToMesh> meshMapper =
      ::ttnn::distributed::create_mesh_mapper(meshDevice, meshMapperConfig);
  ::ttnn::Tensor out = isAlreadyFullyReplicatedOnMesh(input, meshDevice,
                                                      meshMapperConfig)
                           ? input
                           : ::ttnn::distributed::distribute_tensor(input,
                                                                    *meshMapper);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::ccl
