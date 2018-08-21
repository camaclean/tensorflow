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
add_custom_target(tf_extension_ops)
function(AddUserOps)
  cmake_parse_arguments(_AT "" "" "TARGET;SOURCES;GPUSOURCES;DEPENDS;DISTCOPY;LIBS;LIBNAME" ${ARGN})
  if (tensorflow_ENABLE_GPU AND _AT_GPUSOURCES)
    # if gpu build is enabled and we have gpu specific code,
    # hint to cmake that this needs to go to nvcc
    set (gpu_source ${_AT_GPUSOURCES})
    set (gpu_lib "${_AT_TARGET}_gpu")
    set_source_files_properties(${gpu_source} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
    cuda_compile(gpu_lib ${gpu_source})
  endif()
  if(NOT _AT_LIBNAME)
    if(WIN32)
      set(_AT_LIBNAME "${_AT_TARGET}.pyd")
    else()
      set(_AT_LIBNAME "${_AT_TARGET}.so")
    endif()
  endif()
  
  # create shared library from source and cuda obj
  add_library(${_AT_TARGET} SHARED ${_AT_SOURCES} ${gpu_lib})
  target_link_libraries(${_AT_TARGET} ${_AT_LIBS})
  set(tf_contrib_ops ${tf_contrib_ops} ${_AT_TARGET} PARENT_SCOPE)
  set_target_properties(${_AT_TARGET} PROPERTIES
     VERSION ${TENSORFLOW_LIB_VERSION}
     SOVERSION ${TENSORFLOW_LIB_SOVERSION}
  )
  install(TARGETS ${_AT_TARGET}
    EXPORT TensorflowContribTargets
    LIBRARY DESTINATION "lib${LIBSUFFIX}/tensorflow/contrib")
  set_target_properties(${_AT_TARGET}
    PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib${LIBSUFFIX}/tensorflow/contrib")
  if (tensorflow_ENABLE_GPU AND _AT_GPUSOURCES)
      # some ops call out to cuda directly; need to link libs for the cuda dlls
      target_link_libraries(${_AT_TARGET} ${CUDA_LIBRARIES})
  endif()
  if(WIN32)
    target_link_libraries(${_AT_TARGET} ${pywrap_tensorflow_lib})
    if (_AT_DISTCOPY)
        add_custom_command(TARGET ${_AT_TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${_AT_TARGET}> ${_AT_DISTCOPY}/)
    endif()
  else()
    if (_AT_DISTCOPY)
        add_custom_command(TARGET ${_AT_TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${_AT_TARGET}> ${_AT_DISTCOPY}/${_AT_LIBNAME})
    endif()
  endif()
  if (_AT_DEPENDS)
    add_dependencies(${_AT_TARGET} ${_AT_DEPENDS} tf_core_framework tensorflow_contrib_protos)
  endif()
  # make sure TF_COMPILE_LIBRARY is not defined for this target
  get_target_property(target_compile_flags  ${_AT_TARGET} COMPILE_FLAGS)
  if(target_compile_flags STREQUAL "target_compile_flags-NOTFOUND")
    if(WIN32)
      set(target_compile_flags "/UTF_COMPILE_LIBRARY")
    else(WIN32)
      set(target_compile_flags "-UTF_COMPILE_LIBRARY")
    endif(WIN32)
  else()
    if(WIN32)
      set(target_compile_flags "${target_compile_flags} /UTF_COMPILE_LIBRARY")
    else(WIN32)
      set(target_compile_flags "${target_compile_flags} -UTF_COMPILE_LIBRARY")
    endif(WIN32)
  endif()
  set_target_properties(${_AT_TARGET} PROPERTIES COMPILE_FLAGS ${target_compile_flags})
  add_dependencies(tf_extension_ops ${_AT_TARGET})
endfunction(AddUserOps)

function(RELATIVE_PROTOBUF_GENERATE_CPP SRCS HDRS ROOT_DIR)
  if(NOT ARGN)
    message(SEND_ERROR "Error: RELATIVE_PROTOBUF_GENERATE_CPP() called without any proto files")
    return()
  endif()

  set(${SRCS})
  set(${HDRS})
  foreach(FIL ${ARGN})
    set(ABS_FIL ${ROOT_DIR}/${FIL})
    get_filename_component(FIL_WE ${FIL} NAME_WE)
    get_filename_component(FIL_DIR ${ABS_FIL} PATH)
    file(RELATIVE_PATH REL_DIR ${ROOT_DIR} ${FIL_DIR})

    list(APPEND ${SRCS} "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.pb.cc")
    list(APPEND ${HDRS} "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.pb.h")

    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.pb.cc"
             "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.pb.h"
      COMMAND  ${PROTOBUF_PROTOC_EXECUTABLE}
      ARGS --cpp_out  ${CMAKE_CURRENT_BINARY_DIR} -I ${ROOT_DIR} ${ABS_FIL} -I ${PROTOBUF_INCLUDE_DIRS}
      DEPENDS ${ABS_FIL}
      COMMENT "Running C++ protocol buffer compiler on ${FIL}"
      VERBATIM )
  endforeach()

  set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
endfunction()

add_custom_target(python_library_symlinks)
function(PYTHON_SYMLINK_CUSTOM_OPS_LIB TGT LOC LIBNAME)
  if($<TARGET_PROPERTY:${TGT},IMPORTED>)
    get_property(${TGT}_lib_location TARGET ${TGT} PROPERTY LOCATION)
  else()
    set(${TGT}_lib_location "${CMAKE_INSTALL_PREFIX}/lib${LIBSUFFIX}/tensorflow/contrib/${CMAKE_SHARED_LIBRARY_PREFIX}${TGT}${CMAKE_SHARED_LIBRARY_SUFFIX}")
  endif()
  file(MAKE_DIRECTORY "${LOC}")
  add_custom_command(OUTPUT "${LOC}/${LIBNAME}"
                     COMMAND ${CMAKE_COMMAND} -E create_symlink ${${TGT}_lib_location} "${LOC}/${LIBNAME}"
                     DEPENDS ${TGT}
                     COMMENT "Installing contrib library symlink ${LOC}/${LIBNAME}")

  add_custom_target(${TGT}_python_symlink ALL DEPENDS "${LOC}/${LIBNAME}")
  add_dependencies(python_library_symlinks ${TGT}_python_symlink)
  #install(FILES "${LOC}/${LIBNAME}" DESTINATION "lib${LIBSUFFIX}/tensorflow/test")
endfunction()


# NOTE(mrry): Avoid regenerating the tensorflow/core protos because this
# can cause benign-but-failing-on-Windows-due-to-file-locking conflicts
# when two rules attempt to generate the same file.
file(GLOB_RECURSE tf_contrib_protos_cc_srcs RELATIVE ${tensorflow_source_dir}
    "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/proto/*.proto"
    "${tensorflow_source_dir}/tensorflow/contrib/decision_trees/proto/*.proto"
    "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/proto/*.proto"
    "${tensorflow_source_dir}/tensorflow/contrib/session_bundle/*.proto"
    "${tensorflow_source_dir}/tensorflow/contrib/tensorboard/*.proto"
    "${tensorflow_source_dir}/tensorflow/contrib/tpu/profiler/*.proto"
    "${tensorflow_source_dir}/tensorflow/contrib/training/*.proto"
    "${tensorflow_source_dir}/tensorflow/contrib/mpi/*.proto"
)

#file(GLOB_RECURSE tf_contrib_decision_trees_protos_cc_srcs RELATIVE ${tensorflow_source_dir}
#    "${tensorflow_source_dir}/tensorflow/contrib/decision_trees/proto/*.proto"
#)
#RELATIVE_PROTOBUF_GENERATE_CPP(DT_PROTO_SRCS DT_PROTO_HDRS
#    ${tensorflow_source_dir} ${tf_contrib_decision_trees_protos_cc_srcs}
#)
#add_library(decision_trees_protos SHARED ${DT_PROTO_SRCS} ${DT_PROTO_HDRS})
#
#file(GLOB_RECURSE tf_contrib_tensor_forest_protos_cc_srcs RELATIVE ${tensorflow_source_dir}
#    "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/proto/*.proto"
#)
#RELATIVE_PROTOBUF_GENERATE_CPP(T_F_PROTO_SRCS T_F_PROTO_HDRS
#    ${tensorflow_source_dir} ${tf_contrib_tensor_forest_protos_cc_srcs}
#)
#add_library(tensor_forest_protos SHARED ${T_F_PROTO_SRCS} ${T_F_PROTO_HDRS})
#target_link_libraries(tensor_forest_protos decision_trees_protos)

RELATIVE_PROTOBUF_GENERATE_CPP(CONTRIB_PROTO_SRCS CONTRIB_PROTO_HDRS
    ${tensorflow_source_dir} ${tf_contrib_protos_cc_srcs}
)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/contrib/boosted_trees/proto
        DESTINATION include/tensorflow/contrib/boosted_trees
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/resources
        DESTINATION include/tensorflow/contrib/boosted_trees
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/lib/utils
        DESTINATION include/tensorflow/contrib/boosted_trees
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/lib/trees
        DESTINATION include/tensorflow/contrib/boosted_trees
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/lib/quantiles
        DESTINATION include/tensorflow/contrib/boosted_trees/lib
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/lib/models
        DESTINATION include/tensorflow/contrib/boosted_trees/lib
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/lib/learner
        DESTINATION include/tensorflow/contrib/boosted_trees/lib
        FILES_MATCHING PATTERN "*.h")

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/contrib/decision_trees/proto
        DESTINATION include/tensorflow/contrib/decision_trees
        FILES_MATCHING PATTERN "*.h")

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/contrib/tensor_forest/proto
        DESTINATION include/tensorflow/contrib/tensor_forest
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/kernels
        DESTINATION include/tensorflow/contrib/tensor_forest
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/kernels/v4
        DESTINATION include/tensorflow/contrib/tensor_forest/kernels
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/hybrid/core/ops
        DESTINATION include/tensorflow/contrib/tensor_forest/hybrid/core
        FILES_MATCHING PATTERN "*.h")

install(DIRECTORY ${tensorflow_source_dir}/tensorflow/contrib/session_bundle
        DESTINATION include/tensorflow/contrib
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/contrib/session_bundle
        DESTINATION include/tensorflow/contrib
        FILES_MATCHING PATTERN "*.h")

install(DIRECTORY ${tensorflow_source_dir}/tensorflow/contrib/tensorboard/db
        DESTINATION include/tensorflow/contrib/tensorboard
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/contrib/tensorboard
        DESTINATION include/tensorflow/contrib
        FILES_MATCHING PATTERN "*.h")

install(DIRECTORY ${tensorflow_source_dir}/tensorflow/contrib/tpu/profiler
        DESTINATION include/tensorflow/contrib/tpu
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/contrib/tpu/profiler
        DESTINATION include/tensorflow/contrib/tpu
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/contrib/tpu/proto
        DESTINATION include/tensorflow/contrib/tpu
        FILES_MATCHING PATTERN "*.h")

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/training
        DESTINATION include/tensorflow/contrib
        FILES_MATCHING PATTERN "*.h")

install(DIRECTORY ${tensorflow_source_dir}/tensorflow/contrib/mpi
        DESTINATION include/tensorflow/contrib
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/contrib/mpi
        DESTINATION include/tensorflow/contrib
        FILES_MATCHING PATTERN "*.h")

add_library(tensorflow_contrib_protos SHARED ${CONTRIB_PROTO_SRCS} ${CONTRIB_PROTO_HDRS})
set_target_properties(tensorflow_contrib_protos PROPERTIES
    VERSION ${TENSORFLOW_LIB_VERSION}
    SOVERSION ${TENSORFLOW_LIB_SOVERSION}
)
target_link_libraries(tensorflow_contrib_protos tensorflow_protos)
#list(APPEND tf_contrib_ops tensorflow_contrib_protos)
install(TARGETS tensorflow_contrib_protos
    EXPORT TensorflowContribTargets
    LIBRARY DESTINATION "lib${LIBSUFFIX}"
)

set(tf_op_lib_names
    "audio_ops"
    "array_ops"
    "batch_ops"
    "bitwise_ops"
    "boosted_trees_ops"
    "candidate_sampling_ops"
    "checkpoint_ops"
    "control_flow_ops"
    "ctc_ops"
    "cudnn_rnn_ops"
    "data_flow_ops"
    "dataset_ops"
    "decode_proto_ops"
    "encode_proto_ops"
    "functional_ops"
    "image_ops"
    "io_ops"
    "linalg_ops"
    "list_ops"
    "lookup_ops"
    "logging_ops"
    "manip_ops"
    "math_ops"
    "nn_ops"
    "no_op"
    "parsing_ops"
    "random_ops"
    "remote_fused_graph_ops"
    "resource_variable_ops"
    "rpc_ops"
    "script_ops"
    "sdca_ops"
    "set_ops"
    "sendrecv_ops"
    "sparse_ops"
    "spectral_ops"
    "state_ops"
    "stateless_random_ops"
    "string_ops"
    "summary_ops"
    "training_ops"
)

foreach(tf_op_lib_name ${tf_op_lib_names})
    ########################################################
    # tf_${tf_op_lib_name} library
    ########################################################
    file(GLOB tf_${tf_op_lib_name}_srcs
        "${tensorflow_source_dir}/tensorflow/core/ops/${tf_op_lib_name}.cc"
    )

    add_library(tf_${tf_op_lib_name} OBJECT ${tf_${tf_op_lib_name}_srcs})

    add_dependencies(tf_${tf_op_lib_name} tf_core_framework)
endforeach()

function(GENERATE_CONTRIB_OP_LIBRARY op_lib_name cc_srcs)
    add_library(tf_contrib_${op_lib_name}_ops OBJECT ${cc_srcs})
    add_dependencies(tf_contrib_${op_lib_name}_ops tf_core_framework)
endfunction()

file(GLOB_RECURSE tensor_forest_hybrid_srcs
     "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/hybrid/core/ops/*.cc"
)

file(GLOB_RECURSE tpu_ops_srcs
     "${tensorflow_source_dir}/tensorflow/contrib/tpu/ops/*.cc"
)

#GENERATE_CONTRIB_OP_LIBRARY(batch "${tensorflow_source_dir}/tensorflow/contrib/batching/ops/batch_ops.cc")
#GENERATE_CONTRIB_OP_LIBRARY(boosted_trees_ensemble_optimizer "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/ops/ensemble_optimizer_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(boosted_trees_model "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/ops/model_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(boosted_trees_split_handler "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/ops/split_handler_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(boosted_trees_training "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/ops/training_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(boosted_trees_prediction "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/ops/prediction_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(boosted_trees_quantiles "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/ops/quantile_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(boosted_trees_stats_accumulator "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/ops/stats_accumulator_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(bigtable "${tensorflow_source_dir}/tensorflow/contrib/bigtable/ops/bigtable_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(coder "${tensorflow_source_dir}/tensorflow/contrib/coder/ops/coder_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(data_dataset "${tensorflow_source_dir}/tensorflow/contrib/data/ops/dataset_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(factorization_clustering "${tensorflow_source_dir}/tensorflow/contrib/factorization/ops/clustering_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(factorization_factorization "${tensorflow_source_dir}/tensorflow/contrib/factorization/ops/factorization_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(framework_variable "${tensorflow_source_dir}/tensorflow/contrib/framework/ops/variable_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(input_pipeline "${tensorflow_source_dir}/tensorflow/contrib/input_pipeline/ops/input_pipeline_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(image "${tensorflow_source_dir}/tensorflow/contrib/image/ops/image_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(image_distort_image "${tensorflow_source_dir}/tensorflow/contrib/image/ops/distort_image_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(image_sirds "${tensorflow_source_dir}/tensorflow/contrib/image/ops/single_image_random_dot_stereograms_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(layers_sparse_feature_cross "${tensorflow_source_dir}/tensorflow/contrib/layers/ops/sparse_feature_cross_op.cc")
GENERATE_CONTRIB_OP_LIBRARY(memory_stats "${tensorflow_source_dir}/tensorflow/contrib/memory_stats/ops/memory_stats_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(nccl "${tensorflow_source_dir}/tensorflow/contrib/nccl/ops/nccl_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(periodic_resample "${tensorflow_source_dir}/tensorflow/contrib/periodic_resample/ops/array_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(nearest_neighbor "${tensorflow_source_dir}/tensorflow/contrib/nearest_neighbor/ops/nearest_neighbor_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(resampler "${tensorflow_source_dir}/tensorflow/contrib/resampler/ops/resampler_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(rnn_gru "${tensorflow_source_dir}/tensorflow/contrib/rnn/ops/gru_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(rnn_lstm "${tensorflow_source_dir}/tensorflow/contrib/rnn/ops/lstm_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(seq2seq_beam_search "${tensorflow_source_dir}/tensorflow/contrib/seq2seq/ops/beam_search_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(tensor_forest "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/ops/tensor_forest_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(tensor_forest_hybrid "${tensor_forest_hybrid_srcs}")
GENERATE_CONTRIB_OP_LIBRARY(tensor_forest_model "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/ops/model_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(tensor_forest_stats "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/ops/stats_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(text_skip_gram "${tensorflow_source_dir}/tensorflow/contrib/text/ops/skip_gram_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(tpu "${tpu_ops_srcs}")
GENERATE_CONTRIB_OP_LIBRARY(bigquery_reader "${tensorflow_source_dir}/tensorflow/contrib/cloud/ops/bigquery_reader_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(gcs_config "${tensorflow_source_dir}/tensorflow/contrib/cloud/ops/gcs_config_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(reduce_slice "${tensorflow_source_dir}/tensorflow/contrib/reduce_slice_ops/ops/reduce_slice_ops.cc")

set(tf_nearest_neighbor_srcs
    "${tensorflow_source_dir}/tensorflow/contrib/nearest_neighbor/kernels/heap.h"
    "${tensorflow_source_dir}/tensorflow/contrib/nearest_neighbor/kernels/hyperplane_lsh_probes.h"
    "${tensorflow_source_dir}/tensorflow/contrib/nearest_neighbor/kernels/hyperplane_lsh_probes.cc"
    $<TARGET_OBJECTS:tf_contrib_nearest_neighbor_ops>
)

AddUserOps(TARGET nearest_neighbor_ops
    SOURCES "${tf_nearest_neighbor_srcs}"
    DEPENDS tf_contrib_nearest_neighbor_ops)

set(tf_gru_srcs
    "${tensorflow_source_dir}/tensorflow/contrib/rnn/kernels/blas_gemm.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/rnn/kernels/blas_gemm.h"
    "${tensorflow_source_dir}/tensorflow/contrib/rnn/kernels/gru_ops.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/rnn/kernels/gru_ops.h"
    $<TARGET_OBJECTS:tf_contrib_rnn_gru_ops>
)
set(tf_gru_gpu_srcs
    "${tensorflow_source_dir}/tensorflow/contrib/rnn/kernels/gru_ops_gpu.cu.cc"
)

set(tf_lstm_srcs
    "${tensorflow_source_dir}/tensorflow/contrib/rnn/kernels/blas_gemm.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/rnn/kernels/blas_gemm.h"
    "${tensorflow_source_dir}/tensorflow/contrib/rnn/kernels/lstm_ops.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/rnn/kernels/lstm_ops.h"
    $<TARGET_OBJECTS:tf_contrib_rnn_lstm_ops>
)
set(tf_lstm_gpu_srcs
    "${tensorflow_source_dir}/tensorflow/contrib/rnn/kernels/lstm_ops_gpu.cu.cc"
)

AddUserOps(TARGET gru_ops
    SOURCES "${tf_gru_srcs}"
    GPUSOURCES ${tf_gru_gpu_srcs}
    DEPENDS tf_contrib_rnn_gru_ops)

AddUserOps(TARGET lstm_ops
    SOURCES "${tf_lstm_srcs}"
    GPUSOURCES ${tf_lstm_gpu_srcs}
    DEPENDS tf_contrib_rnn_lstm_ops)

set(tf_beam_search_srcs
    "${tensorflow_source_dir}/tensorflow/contrib/seq2seq/kernels/beam_search_ops.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/seq2seq/kernels/beam_search_ops.h"
    $<TARGET_OBJECTS:tf_contrib_seq2seq_beam_search_ops>
)

set(tf_beam_search_gpu_srcs
    "${tensorflow_source_dir}/tensorflow/contrib/seq2seq/kernels/beam_search_ops_gpu.cu.cc"
)

AddUserOps(TARGET beam_search_ops
    SOURCES "${tf_beam_search_srcs}"
    GPUSOURCES ${tf_beam_search_gpu_srcs}
    DEPENDS tf_contrib_seq2seq_beam_search_ops)

set(distort_image_ops_srcs
    "${tensorflow_source_dir}/tensorflow/contrib/image/kernels/adjust_hsv_in_yiq_op.cc"
    $<TARGET_OBJECTS:tf_contrib_image_distort_image_ops>
)

set(distort_image_ops_gpu_srcs
    "${tensorflow_source_dir}/tensorflow/contrib/image/kernels/adjust_hsv_in_yiq_op_gpu.cu.cc"
)

AddUserOps(TARGET distort_image_ops
    SOURCES "${distort_image_ops_srcs}"
    GPUSOURCES "${distort_image_ops_gpu_srcs}"
    DEPENDS tf_contrib_image_distort_image_ops)

AddUserOps(TARGET tpu_ops
    SOURCES $<TARGET_OBJECTS:tf_contrib_tpu_ops>
    DEPENDS tf_contrib_tpu_ops)

set(tf_skip_gram_srcs
    "${tensorflow_source_dir}/tensorflow/contrib/text/kernels/skip_gram_kernels.cc"
    $<TARGET_OBJECTS:tf_contrib_text_skip_gram_ops>
)

AddUserOps(TARGET skip_gram_ops
    SOURCES "${tf_skip_gram_srcs}"
    DEPENDS tf_contrib_text_skip_gram_ops)

add_library(tree_utils OBJECT
    "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/kernels/tree_utils.cc"
)
add_dependencies(tree_utils tf_core_framework tensorflow_contrib_protos)

set(tf_tensor_forest_v2_srcs
    "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/kernels/reinterpret_string_to_float_op.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/kernels/scatter_add_ndim_op.cc"
    $<TARGET_OBJECTS:tf_contrib_tensor_forest_ops>
    $<TARGET_OBJECTS:tree_utils>
)

AddUserOps(TARGET tensor_forest_ops
    SOURCES "${tf_tensor_forest_v2_srcs}"
    DEPENDS tf_contrib_tensor_forest_ops tree_utils)

add_library(tensor_forest_v4_common OBJECT
    "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/kernels/v4/input_data.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/kernels/v4/decision-tree-resource.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/kernels/v4/leaf_model_operators.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/kernels/v4/decision_node_evaluator.cc"
)
add_dependencies(tensor_forest_v4_common tf_core_framework tensorflow_contrib_protos)

set(tf_tensor_forest_model_srcs
    $<TARGET_OBJECTS:tf_contrib_tensor_forest_model_ops>
    "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/kernels/model_ops.cc"
    $<TARGET_OBJECTS:tensor_forest_v4_common>
    $<TARGET_OBJECTS:tree_utils>
)

AddUserOps(TARGET tensor_forest_model_ops
    SOURCES "${tf_tensor_forest_model_srcs}"
    DEPENDS tf_contrib_tensor_forest_model_ops tree_utils tensor_forest_v4_common
    LIBS tensorflow_contrib_protos
)

set(tf_tensor_forest_stats_srcs
    $<TARGET_OBJECTS:tf_contrib_tensor_forest_stats_ops>
    "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/kernels/stats_ops.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/kernels/v4/params.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/kernels/v4/fertile-stats-resource.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/kernels/v4/split_collection_operators.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/kernels/v4/grow_stats.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/kernels/v4/stat_utils.cc"
    $<TARGET_OBJECTS:tensor_forest_v4_common>
    $<TARGET_OBJECTS:tree_utils>
)

AddUserOps(TARGET tensor_forest_stats_ops
    SOURCES "${tf_tensor_forest_stats_srcs}"
    DEPENDS tf_contrib_tensor_forest_stats_ops tree_utils tensor_forest_v4_common
    DISTCOPY ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/tensor_forest/python/ops/
    LIBS tensorflow_contrib_protos
)

set(tf_tensor_forest_hybrid_srcs
    $<TARGET_OBJECTS:tf_contrib_tensor_forest_hybrid_ops>
    $<TARGET_OBJECTS:tree_utils>
)

AddUserOps(TARGET tensor_forest_training_ops
    SOURCES "${tf_tensor_forest_hybrid_srcs}"
    DEPENDS tf_contrib_tensor_forest_hybrid_ops tree_utils)

set(tf_reduce_slice_ops_srcs
    $<TARGET_OBJECTS:tf_contrib_reduce_slice_ops>
    "${tensorflow_source_dir}/tensorflow/contrib/reduce_slice_ops/kernels/reduce_slice_ops.cc"
)
set(tf_reduce_slice_ops_gpu_srcs
    "${tensorflow_source_dir}/tensorflow/contrib/reduce_slice_ops/kernels/reduce_slice_ops_gpu.cu.cc"
)

AddUserOps(TARGET reduce_slice_ops
    SOURCES "${tf_reduce_slice_ops_srcs}"
    GPUSOURCES ${tf_reduce_slice_ops_gpu_srcs}
    DEPENDS tf_contrib_reduce_slice_ops)

set(tf_resampler_srcs
    "${tensorflow_source_dir}/tensorflow/contrib/resampler/kernels/resampler_ops.cc"
    $<TARGET_OBJECTS:tf_contrib_resampler_ops>
)
set(tf_resampler_gpu_srcs
    "${tensorflow_source_dir}/tensorflow/contrib/resampler/kernels/resampler_ops_gpu.cu.cc"
)

AddUserOps(TARGET resampler_ops
    SOURCES "${tf_resampler_srcs}"
    GPUSOURCES ${tf_resampler_gpu_srcs}
    DEPENDS tf_contrib_resampler_ops)

if(NOT WIN32 AND tensorflow_ENABLE_GPU AND NCCL_LIBRARY)
    set(tf_nccl_srcs
        "${tensorflow_source_dir}/tensorflow/contrib/nccl/kernels/nccl_manager.cc"
        "${tensorflow_source_dir}/tensorflow/contrib/nccl/kernels/nccl_ops.cc"
        $<TARGET_OBJECTS:tf_contrib_nccl_ops>
    )
    AddUserOps(TARGET nccl_ops
        SOURCES "${tf_nccl_srcs}"
        DEPENDS tf_contrib_nccl_ops
        LIBS "${NCCL_LIBRARY}" ${CUDA_LIBRARIES})
endif()

set(tf_memory_stats_srcs
    "${tensorflow_source_dir}/tensorflow/contrib/memory_stats/kernels/memory_stats_ops.cc"
    $<TARGET_OBJECTS:tf_contrib_memory_stats_ops>
)

AddUserOps(TARGET memory_stats_ops
    SOURCES "${tf_memory_stats_srcs}"
    DEPENDS tf_contrib_memory_stats_ops)

AddUserOps(TARGET sparse_feature_cross_op
    SOURCES $<TARGET_OBJECTS:tf_contrib_layers_sparse_feature_cross_ops>
    DEPENDS tf_contrib_layers_sparse_feature_cross_ops)

AddUserOps(TARGET input_pipeline_ops
    SOURCES $<TARGET_OBJECTS:tf_contrib_input_pipeline_ops>
    DEPENDS tf_contrib_input_pipeline_ops)

set(tf_image_srcs
    "${tensorflow_source_dir}/tensorflow/contrib/image/kernels/bipartite_match_op.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/image/kernels/image_ops.cc"
    $<TARGET_OBJECTS:tf_contrib_image_ops>
)

set(tf_image_gpu_srcs
    "${tensorflow_source_dir}/tensorflow/contrib/image/kernels/image_ops_gpu.cu.cc"
)

AddUserOps(TARGET image_ops
    SOURCES "${tf_image_srcs}"
    GPUSOURCES ${tf_image_gpu_srcs}
    DEPENDS tf_contrib_image_ops)

set(tf_single_image_random_dot_stereograms_srcs
    "${tensorflow_source_dir}/tensorflow/contrib/image/kernels/single_image_random_dot_stereograms_ops.cc"
    $<TARGET_OBJECTS:tf_contrib_image_sirds_ops>
)

AddUserOps(TARGET single_image_random_dot_stereograms
    SOURCES "${tf_single_image_random_dot_stereograms_srcs}"
    DEPENDS tf_contrib_image_sirds_ops)

set(tf_framework_variable_srcs
    "${tensorflow_source_dir}/tensorflow/contrib/framework/kernels/zero_initializer_op.cc"
    $<TARGET_OBJECTS:tf_contrib_framework_variable_ops>
)

set(tf_framework_variable_gpu_srcs
    "${tensorflow_source_dir}/tensorflow/contrib/framework/kernels/zero_initializer_op_gpu.cu.cc"
)

AddUserOps(TARGET variable_ops
    SOURCES "${tf_framework_variable_srcs}"
    GPUSOURCES ${tf_framework_variable_gpu_srcs}
    DEPENDS tf_contrib_framework_variable_ops)

set(tf_factorization_clustering_srcs
    "${tensorflow_source_dir}/tensorflow/contrib/factorization/kernels/clustering_ops.cc"
    $<TARGET_OBJECTS:tf_contrib_factorization_clustering_ops>
)

AddUserOps(TARGET clustering_ops
    SOURCES "${tf_factorization_clustering_srcs}"
    DEPENDS tf_contrib_factorization_clustering_ops)

set(tf_factorization_srcs
    "${tensorflow_source_dir}/tensorflow/contrib/factorization/kernels/masked_matmul_ops.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/factorization/kernels/wals_solver_ops.cc"
    $<TARGET_OBJECTS:tf_contrib_factorization_factorization_ops>
)

AddUserOps(TARGET factorization_ops
    SOURCES "${tf_factorization_srcs}"
    DEPENDS tf_contrib_factorization_factorization_ops)

set(tf_boosted_trees_utils_srcs
    "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/lib/utils/batch_features.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/lib/utils/dropout_utils.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/lib/utils/examples_iterable.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/lib/utils/parallel_for.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/lib/utils/sparse_column_iterable.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/lib/utils/tensor_utils.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/lib/learner/common/partitioners/example_partitioner.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/lib/models/multiple_additive_trees.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/lib/trees/decision_tree.cc"
)
add_library(boosted_trees_utils SHARED ${tf_boosted_trees_utils_srcs})
set_target_properties(boosted_trees_utils PROPERTIES
    VERSION ${TENSORFLOW_LIB_VERSION}
    SOVERSION ${TENSORFLOW_LIB_SOVERSION}
)
target_link_libraries(boosted_trees_utils tensorflow_contrib_protos)
list(APPEND tf_contrib_ops boosted_trees_utils)
install(TARGETS boosted_trees_utils
    EXPORT TensorflowContribTargets
    LIBRARY DESTINATION "lib${LIBSUFFIX}/tensorflow/contrib")

set(tf_boosted_trees_srcs
    "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/kernels/model_ops.cc"
    $<TARGET_OBJECTS:tf_contrib_boosted_trees_model_ops>
    "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/kernels/split_handler_ops.cc"
    $<TARGET_OBJECTS:tf_contrib_boosted_trees_split_handler_ops>
    "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/kernels/training_ops.cc"
    $<TARGET_OBJECTS:tf_contrib_boosted_trees_training_ops>
    "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/kernels/prediction_ops.cc"
    $<TARGET_OBJECTS:tf_contrib_boosted_trees_prediction_ops>
    "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/kernels/quantile_ops.cc"
    $<TARGET_OBJECTS:tf_contrib_boosted_trees_quantiles_ops>
    "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/kernels/stats_accumulator_ops.cc"
    $<TARGET_OBJECTS:tf_contrib_boosted_trees_stats_accumulator_ops>
)

AddUserOps(TARGET boosted_trees_ops
    SOURCES "${tf_boosted_trees_srcs}"
    DEPENDS tf_contrib_boosted_trees_model_ops
    LIBS tensorflow_contrib_protos boosted_trees_utils
)

set(tf_bigtable_srcs
    "${tensorflow_source_dir}/tensorflow/contrib/bigtable/kernels/bigtable_scan_dataset_op.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/bigtable/kernels/bigtable_sample_keys_dataset_op.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/bigtable/kernels/bigtable_sample_key_pairs_dataset_op.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/bigtable/kernels/bigtable_range_key_dataset_op.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/bigtable/kernels/bigtable_range_helpers_test.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/bigtable/kernels/bigtable_range_helpers.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/bigtable/kernels/bigtable_prefix_key_dataset_op.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/bigtable/kernels/bigtable_lookup_dataset_op.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/bigtable/kernels/bigtable_lib.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/bigtable/kernels/bigtable_kernels.cc"
    $<TARGET_OBJECTS:tf_contrib_bigtable_ops>
)

AddUserOps(TARGET bigtable_ops
    SOURCES "${tf_bigtable_srcs}"
    DEPENDS tf_contrib_bigtable_ops
    LIBS bigtable_client bigtable_protos) # whole-archive protos?

set(tf_coder_srcs
    "${tensorflow_source_dir}/tensorflow/contrib/coder/kernels/range_coder_ops_util.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/coder/kernels/range_coder_ops.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/coder/kernels/range_coder.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/coder/kernels/pmf_to_cdf_op.cc"
    $<TARGET_OBJECTS:tf_contrib_coder_ops>
)

AddUserOps(TARGET coder_ops
    SOURCES "${tf_coder_srcs}"
    DEPENDS tf_contrib_coder_ops
)

set(tf_dataset_srcs
    "${tensorflow_source_dir}/tensorflow/contrib/data/kernels/unique_dataset_op.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/data/kernels/threadpool_dataset_op.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/data/kernels/prefetching_kernels.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/data/kernels/ignore_errors_dataset_op.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/data/kernels/directed_interleave_dataset_op.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/data/kernels/csv_dataset_op.cc"
    $<TARGET_OBJECTS:tf_contrib_data_dataset_ops>
)

AddUserOps(TARGET dataset_ops
    SOURCES "${tf_dataset_srcs}"
    DEPENDS tf_contrib_data_dataset_ops
)

set(tf_periodic_resample_srcs
    "${tensorflow_source_dir}/tensorflow/contrib/periodic_resample/kernels/periodic_resample_op.cc"
    $<TARGET_OBJECTS:tf_contrib_periodic_resample_ops>
)

AddUserOps(TARGET periodic_resample
    SOURCES "${tf_periodic_resample_srcs}"
    DEPENDS tf_contrib_periodic_resample_ops
)

# Create TensorflowConfig.cmake
EXPORT(TARGETS ${tf_contrib_ops} tensorflow_contrib_protos FILE "${CMAKE_CURRENT_BINARY_DIR}/TensorflowContribTargets.cmake")
INSTALL(EXPORT TensorflowContribTargets DESTINATION "share/tensorflow/cmake")

########################################################
# tf_user_ops library
########################################################
file(GLOB_RECURSE tf_user_ops_srcs
    "${tensorflow_source_dir}/tensorflow/core/user_ops/*.cc"
)

add_library(tf_user_ops OBJECT ${tf_user_ops_srcs})

add_dependencies(tf_user_ops tf_core_framework)

########################################################
# tf_core_ops library
########################################################
file(GLOB_RECURSE tf_core_ops_srcs
    "${tensorflow_source_dir}/tensorflow/core/ops/*.h"
    "${tensorflow_source_dir}/tensorflow/core/ops/*.cc"
    "${tensorflow_source_dir}/tensorflow/core/user_ops/*.h"
    "${tensorflow_source_dir}/tensorflow/core/user_ops/*.cc"
)

file(GLOB_RECURSE tf_core_ops_exclude_srcs
    "${tensorflow_source_dir}/tensorflow/core/ops/*test*.h"
    "${tensorflow_source_dir}/tensorflow/core/ops/*test*.cc"
    "${tensorflow_source_dir}/tensorflow/core/ops/*main.cc"
    "${tensorflow_source_dir}/tensorflow/core/user_ops/*test*.h"
    "${tensorflow_source_dir}/tensorflow/core/user_ops/*test*.cc"
    "${tensorflow_source_dir}/tensorflow/core/user_ops/*main.cc"
    "${tensorflow_source_dir}/tensorflow/core/user_ops/*.cu.cc"
)

list(REMOVE_ITEM tf_core_ops_srcs ${tf_core_ops_exclude_srcs})

add_library(tf_core_ops OBJECT ${tf_core_ops_srcs})

add_dependencies(tf_core_ops tf_core_cpu)

########################################################
# tf_debug_ops library
########################################################

file(GLOB tf_debug_ops_srcs
    "${tensorflow_source_dir}/tensorflow/core/ops/debug_ops.cc"
)

add_library(tf_debug_ops OBJECT ${tf_debug_ops_srcs})

add_dependencies(tf_debug_ops tf_core_framework)
