# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
set(tf_tools_proto_text_src_dir "${tensorflow_source_dir}/tensorflow/tools/proto_text")

file(GLOB tf_tools_proto_text_srcs
    "${tf_tools_proto_text_src_dir}/gen_proto_text_functions.cc"
    "${tf_tools_proto_text_src_dir}/gen_proto_text_functions_lib.h"
    "${tf_tools_proto_text_src_dir}/gen_proto_text_functions_lib.cc"
)

set(proto_text "proto_text")

add_executable(${proto_text}
    ${tf_tools_proto_text_srcs}
    $<TARGET_OBJECTS:tf_core_lib>
)
install(TARGETS ${proto_text}
    EXPORT TensorflowToolsTargets
    RUNTIME DESTINATION "libexec/tensorflow")

target_link_libraries(${proto_text} PUBLIC
  ${tensorflow_EXTERNAL_LIBRARIES}
  tensorflow_protos
)

add_dependencies(${proto_text} tf_core_lib)

file(GLOB_RECURSE tf_tools_transform_graph_lib_srcs
    "${tensorflow_source_dir}/tensorflow/tools/graph_transforms/*.h"
    "${tensorflow_source_dir}/tensorflow/tools/graph_transforms/*.cc"
)

file(GLOB_RECURSE tf_tools_transform_graph_lib_exclude_srcs
    "${tensorflow_source_dir}/tensorflow/tools/graph_transforms/*test*.h"
    "${tensorflow_source_dir}/tensorflow/tools/graph_transforms/*test*.cc"
    "${tensorflow_source_dir}/tensorflow/tools/graph_transforms/compare_graphs.cc"
    "${tensorflow_source_dir}/tensorflow/tools/graph_transforms/summarize_graph_main.cc"
    "${tensorflow_source_dir}/tensorflow/tools/graph_transforms/transform_graph_main.cc"
)
list(REMOVE_ITEM tf_tools_transform_graph_lib_srcs ${tf_tools_transform_graph_lib_exclude_srcs})

add_library(tf_tools_transform_graph_lib OBJECT ${tf_tools_transform_graph_lib_srcs})
add_dependencies(tf_tools_transform_graph_lib tf_core_cpu)
add_dependencies(tf_tools_transform_graph_lib tf_core_framework)
add_dependencies(tf_tools_transform_graph_lib tf_core_kernels)
add_dependencies(tf_tools_transform_graph_lib tf_core_lib)
add_dependencies(tf_tools_transform_graph_lib tf_core_ops)

set(transform_graph "transform_graph")

add_executable(${transform_graph}
    "${tensorflow_source_dir}/tensorflow/tools/graph_transforms/transform_graph_main.cc"
)

target_link_libraries(${transform_graph} PUBLIC
  tensorflow
  tensorflow_protos
  ${tf_core_gpu_kernels_lib}
  ${tensorflow_EXTERNAL_LIBRARIES}
)

set(summarize_graph "summarize_graph")

add_executable(${summarize_graph}
    "${tensorflow_source_dir}/tensorflow/tools/graph_transforms/summarize_graph_main.cc"
)

target_link_libraries(${summarize_graph} PUBLIC
  tensorflow
  tensorflow_protos
  ${tf_core_gpu_kernels_lib}
  ${tensorflow_EXTERNAL_LIBRARIES}
)

set(compare_graphs "compare_graphs")

add_executable(${compare_graphs}
    "${tensorflow_source_dir}/tensorflow/tools/graph_transforms/compare_graphs.cc"
)

target_link_libraries(${compare_graphs} PUBLIC
  tensorflow
  tensorflow_protos
  ${tf_core_gpu_kernels_lib}
  ${tensorflow_EXTERNAL_LIBRARIES}
)

set(benchmark_model "benchmark_model")

add_executable(${benchmark_model}
    "${tensorflow_source_dir}/tensorflow/tools/benchmark/benchmark_model.cc"
    "${tensorflow_source_dir}/tensorflow/tools/benchmark/benchmark_model_main.cc"
)

target_link_libraries(${benchmark_model} PUBLIC
  tensorflow
  tensorflow_protos
  ${tf_core_gpu_kernels_lib}
  ${tensorflow_EXTERNAL_LIBRARIES}
)

install(TARGETS ${transform_graph} ${summarize_graph} ${compare_graphs} ${benchmark_model}
    EXPORT TensorflowToolsTargets
    LIBRARY DESTINATION "lib${LIBSUFFIX}"
    RUNTIME DESTINATION "bin")
# Create TensorflowToolsConfig.cmake
EXPORT(TARGETS ${transform_graph} ${summarize_graph} ${compare_graphs} ${benchmark_model} FILE "${CMAKE_CURRENT_BINARY_DIR}/TensorflowToolsTargets.cmake")
INSTALL(EXPORT TensorflowToolsTargets DESTINATION "share/tensorflow/cmake")
