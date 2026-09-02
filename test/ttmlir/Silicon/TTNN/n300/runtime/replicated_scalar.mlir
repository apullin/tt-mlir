// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=1,2" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// Rank-0 scalar fixture where the input is replicated across both chips and
// then aggregated back to the host.
module @Model attributes {} {
  func.func @forward(
      %arg0: tensor<f32> {ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "scalar"})
      -> (tensor<f32> {ttir.name = "Model.output_scalar"}) {
    %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<f32>) -> tensor<f32>
    %1 = "ttir.mesh_shard"(%0) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<replicate>}> : (tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
}

// CHECK: "ttnn.distribute_tensor"
// CHECK-SAME: placements = [<replicate, -1 : i32>, <replicate, -1 : i32>]
