# Minimum CMake required
cmake_minimum_required(VERSION 3.5)

# Project
project(TensorflowMPI C CXX)

# Set C++14 as standard for the whole project
set(CMAKE_CXX_STANDARD 14)

# Actual source is the ../../.. directory
get_filename_component(tf_contrib_source_dir ${TensorflowMPI_SOURCE_DIR} PATH)
get_filename_component(tf_tf_source_dir ${tf_contrib_source_dir} PATH)
get_filename_component(tensorflow_source_dir ${tf_tf_source_dir} PATH)

# [CLEANUP] Not sure if this is needed (copied from Protobuf)
# CMake policies
#cmake_policy(SET CMP0022 NEW)

get_property(LIB64 GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS)

if ("${LIB64}" STREQUAL "TRUE")
    set(LIBSUFFIX 64)
else()
    set(LIBSUFFIX "")
endif()

set(TENSORFLOW_LIB_VERSION 1.4.1)
set(TENSORFLOW_LIB_SOVERSION 1)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(tensorflow_ENABLE_MPI "Enable MPI support" OFF)

set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf REQUIRED)
find_package(Tensorflow REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Eigen3Tensorflow REQUIRED CONFIG)
find_package(jsoncpp REQUIRED)
find_package(nsync REQUIRED)
find_package(Snappy REQUIRED)
find_package(gRPC REQUIRED)
include(FindPythonInterp)

if(tensorflow_ENABLE_MPI)
    find_package(TensorflowMPI REQUIRED)
endif()

include_directories(
    ${TENSORFLOW_INCLUDE_DIR}
    ${tensorflow_source_dir}
    ${EIGEN3_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
)

get_target_property(PROTO_TEXT_CONFIGS proto_text IMPORTED_CONFIGURATIONS)
string(TOUPPER "${CMAKE_BUILD_TYPE}" TRY_CONFIG_TYPE)
list(INSERT PROTO_TEXT_CONFIGS 0 "${TRY_CONFIG_TYPE}")
set(PROTO_TEXT_EXE "PROTO_TEXT_EXE-NOTFOUND")
foreach(PROTO_TEXT_CONFIG ${PROTO_TEXT_CONFIGS})
  if("${PROTO_TEXT_EXE}" STREQUAL "PROTO_TEXT_EXE-NOTFOUND")
    get_target_property(PROTO_TEXT_EXE proto_text IMPORTED_LOCATION_GENTOO)
  endif()
endforeach()

#function(RELATIVE_PROTOBUF_TEXT_GENERATE_CPP SRCS HDRS ROOT_DIR)
#  if(NOT ARGN)
#      message(SEND_ERROR "Error: RELATIVE_PROTOBUF_TEXT_GENERATE_CPP() called without any proto files")
#    return()
#  endif()
#
#  set(${SRCS})
#  set(${HDRS})
#  foreach(FIL ${ARGN})
#    set(ABS_FIL ${ROOT_DIR}/${FIL})
#    get_filename_component(FIL_WE ${FIL} NAME_WE)
#    get_filename_component(FIL_DIR ${ABS_FIL} PATH)
#    file(RELATIVE_PATH REL_DIR ${ROOT_DIR} ${FIL_DIR})
#
#    list(APPEND ${SRCS} "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.pb_text.cc")
#    list(APPEND ${HDRS} "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.pb_text.h")
#
#    add_custom_command(
#      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.pb_text.cc"
#             "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.pb_text.h"
#      COMMAND ${PROTO_TEXT_EXE}
#      ARGS "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}" ${REL_DIR} ${ABS_FIL} "${ROOT_DIR}/tensorflow/tools/proto_text/placeholder.txt"
#      DEPENDS ${ABS_FIL} ${PROTO_TEXT_EXE}
#      COMMENT "Running C++ protocol buffer text compiler (${PROTO_TEXT_EXE}) on ${FIL}"
#      VERBATIM )
#  endforeach()
#
#  set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
#  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
#  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
#endfunction()

if(NOT PYTHON_INCLUDE_DIR)
  set(PYTHON_NOT_FOUND false)
  exec_program("${PYTHON_EXECUTABLE}"
    ARGS "-c \"import distutils.sysconfig; print(distutils.sysconfig.get_python_inc())\""
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    RETURN_VALUE PYTHON_NOT_FOUND)
  if(${PYTHON_NOT_FOUND})
    message(FATAL_ERROR
            "Cannot get Python include directory. Is distutils installed?")
  endif(${PYTHON_NOT_FOUND})
endif(NOT PYTHON_INCLUDE_DIR)
FIND_PACKAGE(PythonLibs)

# 2. Resolve the installed version of NumPy (for numpy/arrayobject.h).
if(NOT NUMPY_INCLUDE_DIR)
  set(NUMPY_NOT_FOUND false)
  exec_program("${PYTHON_EXECUTABLE}"
    ARGS "-c \"import numpy; print(numpy.get_include())\""
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    RETURN_VALUE NUMPY_NOT_FOUND)
  if(${NUMPY_NOT_FOUND})
    message(FATAL_ERROR
            "Cannot get NumPy include directory: Is NumPy installed?")
  endif(${NUMPY_NOT_FOUND})
endif(NOT NUMPY_INCLUDE_DIR)

function(RELATIVE_PROTOBUF_GENERATE_PYTHON ROOT_DIR SRCS)
  if(NOT ARGN)
    message(SEND_ERROR "Error: RELATIVE_PROTOBUF_GENERATE_PYTHON() called without any proto files")
    return()
  endif()

  set(${SRCS})
  foreach(FIL ${ARGN})
    set(ABS_FIL ${ROOT_DIR}/${FIL})
    get_filename_component(FIL_WE ${FIL} NAME_WE)
    get_filename_component(FIL_DIR ${ABS_FIL} PATH)
    file(RELATIVE_PATH REL_DIR ${ROOT_DIR} ${FIL_DIR})

    list(APPEND ${SRCS} "${CMAKE_CURRENT_BINARY_DIR}/tf_python/${REL_DIR}/${FIL_WE}_pb2.py")
    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/tf_python/${REL_DIR}/${FIL_WE}_pb2.py"
      #COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
      COMMAND ${Protobuf_PROTOC_EXECUTABLE}
      ARGS --python_out  ${CMAKE_CURRENT_BINARY_DIR}/tf_python/ -I ${ROOT_DIR} -I ${PROTOBUF_INCLUDE_DIRS} ${ABS_FIL}
      COMMENT "Running Python protocol buffer compiler on ${FIL}"
      VERBATIM )
  endforeach()
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
endfunction()

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
      COMMAND  ${Protobuf_PROTOC_EXECUTABLE}
      ARGS --cpp_out  ${CMAKE_CURRENT_BINARY_DIR} -I ${ROOT_DIR} ${ABS_FIL} -I ${PROTOBUF_INCLUDE_DIRS}
      DEPENDS ${ABS_FIL}
      COMMENT "Running C++ protocol buffer compiler on ${FIL}"
      VERBATIM )
  endforeach()

  set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
endfunction()

FILE(READ python_protos.txt python_protos)
# Convert file contents into a CMake list (where each element in the list is one line of the file)
STRING(REGEX REPLACE ";" "\\\\;" python_protos "${python_protos}")
STRING(REGEX REPLACE "\n" ";" python_protos "${python_protos}")

foreach(python_proto ${python_protos})
  if(NOT python_proto MATCHES "^\#")
    STRING(REGEX REPLACE " *\#.*" "" python_proto "${python_proto}")
    if(NOT EXISTS "${tensorflow_source_dir}/${python_proto}")
      message(SEND_ERROR "Python proto directory not found: ${python_proto}")
    endif()
    file(GLOB_RECURSE tf_python_protos_src RELATIVE ${tensorflow_source_dir}
        "${tensorflow_source_dir}/${python_proto}/*.proto"
    )
    list(APPEND tf_python_protos_srcs ${tf_python_protos_src})
  endif()
endforeach(python_proto)

RELATIVE_PROTOBUF_GENERATE_PYTHON(
    ${tensorflow_source_dir} PYTHON_PROTO_GENFILES ${tf_python_protos_srcs}
)

FILE(READ python_protos_cc.txt python_protos_cc)
# Convert file contents into a CMake list (where each element in the list is one line of the file)
STRING(REGEX REPLACE ";" "\\\\;" python_protos_cc "${python_protos_cc}")
STRING(REGEX REPLACE "\n" ";" python_protos_cc "${python_protos_cc}")

foreach(python_proto_cc ${python_protos_cc})
  if(NOT python_proto_cc MATCHES "^\#")
    STRING(REGEX REPLACE " *\#.*" "" python_proto_cc "${python_proto_cc}")
    if(NOT EXISTS "${tensorflow_source_dir}/${python_proto_cc}")
      message(SEND_ERROR "Python proto CC directory not found: ${python_proto_cc}")
    endif()
    file(GLOB_RECURSE tf_python_protos_cc_src RELATIVE ${tensorflow_source_dir}
        "${tensorflow_source_dir}/${python_proto_cc}/*.proto"
    )
    list(APPEND tf_python_protos_cc_srcs ${tf_python_protos_cc_src})
  endif()
endforeach(python_proto_cc)

RELATIVE_PROTOBUF_GENERATE_CPP(ALL_PROTO_SRCS ALL_PROTO_HDRS
    ${tensorflow_source_dir} ${tf_python_protos_cc_srcs}
)

file(GLOB_RECURSE tf_framework_protos_cc_srcs RELATIVE ${tensorflow_source_dir}
    "${tensorflow_source_dir}/tensorflow/core/profiler/*.proto"
    "${tensorflow_source_dir}/tensorflow/core/*.proto"
    "${tensorflow_source_dir}/tensorflow/core/framework/*.proto"
    "${tensorflow_source_dir}/tensorflow/core/lib/core/error_codes.proto"
    "${tensorflow_source_dir}/tensorflow/core/protobuf/cluster.proto"
    "${tensorflow_source_dir}/tensorflow/core/protobuf/config.proto"
    "${tensorflow_source_dir}/tensorflow/core/protobuf/debug.proto"
    "${tensorflow_source_dir}/tensorflow/core/protobuf/device_properties.proto"
    "${tensorflow_source_dir}/tensorflow/core/protobuf/rewriter_config.proto"
    "${tensorflow_source_dir}/tensorflow/core/protobuf/tensor_bundle.proto"
    "${tensorflow_source_dir}/tensorflow/core/protobuf/saver.proto"
    "${tensorflow_source_dir}/tensorflow/core/util/memmapped_file_system.proto"
    "${tensorflow_source_dir}/tensorflow/core/util/saved_tensor_slice.proto"
)
RELATIVE_PROTOBUF_GENERATE_CPP(FRAMEWORK_PROTO_SRCS FRAMEWORK_PROTO_HDRS
    ${tensorflow_source_dir} ${tf_framework_protos_cc_srcs}
)
set(tf_core_text_protos_cc_srcs
    "tensorflow/core/framework/device_attributes.proto"
    "tensorflow/core/framework/tensor.proto"
    "tensorflow/core/framework/op_def.proto"
)

#RELATIVE_PROTOBUF_TEXT_GENERATE_CPP(CORE_TEXT_PROTO_SRCS CORE_TEXT_PROTO_HDRS
#  ${tensorflow_source_dir} ${tf_core_text_protos_cc_srcs}
#)
#add_custom_target(text_proto_hdrs DEPENDS ${CORE_TEXT_PROTO_HDRS})
#add_dependencies(text_proto_hdrs framework_proto_hdrs)

add_custom_target(framework_proto_hdrs DEPENDS ${FRAMEWORK_PROTO_HDRS})
file(GLOB_RECURSE tf_core_protos_cc_srcs RELATIVE ${tensorflow_source_dir}
    "${tensorflow_source_dir}/tensorflow/core/protobuf/*.proto"
    "${tensorflow_source_dir}/tensorflow/core/lib/*.proto"
)
RELATIVE_PROTOBUF_GENERATE_CPP(CORE_PROTO_SRCS CORE_PROTO_HDRS
    ${tensorflow_source_dir} ${tf_core_protos_cc_srcs}
)
add_custom_target(core_proto_hdrs DEPENDS ${CORE_PROTO_HDRS})
add_dependencies(core_proto_hdrs framework_proto_hdrs text_proto_hdrs)

file(GLOB_RECURSE tf_python_protos_cc_srcs RELATIVE ${tensorflow_source_dir}
    "${tensorflow_source_dir}/tensorflow/python/*.proto"
)
RELATIVE_PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS
    ${tensorflow_source_dir} ${tf_python_protos_cc_srcs}
)

add_library(tf_python_all_protos_cc ${ALL_PROTO_SRCS} ${ALL_PROTO_HDRS})
add_dependencies(tf_python_all_protos_cc core_proto_hdrs text_proto_hdrs)

add_library(tf_python_protos_cc ${PROTO_SRCS} ${PROTO_HDRS})
add_dependencies(tf_python_protos_cc core_proto_hdrs text_proto_hdrs)

# tf_python_touchup_modules adds empty __init__.py files to all
# directories containing Python code, so that Python will recognize
# them as modules.
add_custom_target(tf_python_touchup_modules)

# tf_python_copy_scripts_to_destination copies all Python files
# (including static source and generated protobuf wrappers, but *not*
# generated TensorFlow op wrappers) into tf_python/.
add_custom_target(tf_python_copy_scripts_to_destination DEPENDS tf_python_touchup_modules)

#prevent ninja from using a command that is way too long
set(module_number "0")
# tf_python_srcs contains all static .py files
function(add_python_module MODULE_NAME)
    set(options DONTCOPY)
    cmake_parse_arguments(ADD_PYTHON_MODULE "${options}" "" "" ${ARGN})
    add_custom_target(tf_python_touchup_modules_${module_number})
    if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/tf_python/${MODULE_NAME}/__init__.py")
        add_custom_command(TARGET tf_python_touchup_modules_${module_number} PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/tf_python/${MODULE_NAME}")
        add_custom_command(TARGET tf_python_touchup_modules_${module_number} PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E touch "${CMAKE_CURRENT_BINARY_DIR}/tf_python/${MODULE_NAME}/__init__.py")
    endif()
    add_dependencies(tf_python_touchup_modules tf_python_touchup_modules_${module_number})
    MATH(EXPR module_number_new "${module_number}+1")
    set(module_number "${module_number_new}" PARENT_SCOPE)
    file(GLOB module_python_srcs RELATIVE ${tensorflow_source_dir}
        "${tensorflow_source_dir}/${MODULE_NAME}/*.py"
    )
    if(NOT ${ADD_PYTHON_MODULE_DONTCOPY})
        add_custom_target(tf_python_copy_scripts_to_destination_${module_number})
        foreach(script ${module_python_srcs})
            get_filename_component(REL_DIR ${script} DIRECTORY)
            # NOTE(mrry): This rule may exclude modules that should be part of
            # the distributed PIP package
            # (e.g. tensorflow/contrib/testing/python/framework/util_test.py),
            # so we currently add explicit commands to include those files
            # later on in this script.
            if (NOT "${script}" MATCHES "_test\.py$")
	        add_custom_command(TARGET tf_python_copy_scripts_to_destination_${module_number} PRE_BUILD
                  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tensorflow_source_dir}/${script} ${CMAKE_CURRENT_BINARY_DIR}/tf_python/${script})
                #MATH(EXPR module_source_number_new "${module_source_number}+1")
                #set(module_source_number "${module_source_number_new}" PARENT_SCOPE)
            endif()
        endforeach()
        add_dependencies(tf_python_copy_scripts_to_destination tf_python_copy_scripts_to_destination_${module_number})
    endif()
endfunction()

FILE(READ python_modules.txt python_modules)
# Convert file contents into a CMake list (where each element in the list is one line of the file)
STRING(REGEX REPLACE ";" "\\\\;" python_modules "${python_modules}")
STRING(REGEX REPLACE "\n" ";" python_modules "${python_modules}")

foreach(python_module ${python_modules})
  if(NOT python_module MATCHES "^\#")
    STRING(REGEX REPLACE " *\#.*" "" python_module "${python_module}")
    if(NOT EXISTS "${tensorflow_source_dir}/${python_module}")
      message(SEND_ERROR "Python module not found: ${python_module}")
    endif()
    add_python_module(${python_module})
  endif()
endforeach(python_module)

add_custom_command(TARGET tf_python_touchup_modules PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
    "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/lite")
add_custom_command(TARGET tf_python_touchup_modules PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
    "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/lite/python")
add_custom_command(TARGET tf_python_touchup_modules PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E touch
    "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/lite/python/__init__.py")
add_custom_command(
    TARGET tf_python_copy_scripts_to_destination PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E touch
    ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/lite/python/lite.py)

# Generate the tensorflow.python.platform.build_info module.
if (tensorflow_ENABLE_GPU)
    set(tensorflow_BUILD_INFO_FLAGS --build_config cuda --key_value
	    cuda_version_number=${tensorflow_CUDA_VERSION}
	    cudnn_version_number=${tensorflow_CUDNN_VERSION})
else()
  set(tensorflow_BUILD_INFO_FLAGS --build_config cpu --key_value
    msvcp_dll_name=msvcp140.dll)
endif()

# Generate the tensorflow.python.platform.build_info module.
set(BUILD_INFO_PY "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/python/platform/build_info.py")
add_custom_command(TARGET tf_python_copy_scripts_to_destination PRE_BUILD
  COMMAND ${PYTHON_EXECUTABLE} ${tensorflow_source_dir}/tensorflow/tools/build_info/gen_build_info.py --raw_generate ${BUILD_INFO_PY} ${tensorflow_BUILD_INFO_FLAGS})


file(GLOB_RECURSE tf_user_ops_srcs
    "${tensorflow_source_dir}/tensorflow/core/user_ops/*.cc"
)

add_library(tf_user_ops OBJECT ${tf_user_ops_srcs})
add_dependencies(tf_user_ops tf_python_all_protos_cc)

set(tf_python_op_gen_main_srcs
    "${tensorflow_source_dir}/tensorflow/python/eager/python_eager_op_gen.h"
    "${tensorflow_source_dir}/tensorflow/python/eager/python_eager_op_gen.cc"
    "${tensorflow_source_dir}/tensorflow/python/framework/python_op_gen.cc"
    "${tensorflow_source_dir}/tensorflow/python/framework/python_op_gen.cc"
    "${tensorflow_source_dir}/tensorflow/python/framework/python_op_gen_main.cc"
    "${tensorflow_source_dir}/tensorflow/python/framework/python_op_gen.h"
    "${tensorflow_source_dir}/tensorflow/python/framework/python_op_gen_internal.h"
)

add_library(tf_python_op_gen_main OBJECT ${tf_python_op_gen_main_srcs})
add_dependencies(tf_python_op_gen_main tf_python_protos_cc tf_python_all_protos_cc)

# create directory for ops generated files
set(python_ops_target_dir ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/python/ops)

set(tf_python_ops_generated_files)

set(tf_python_op_lib_names
    ${tf_op_lib_names}
    "user_ops"
)

function(GENERATE_PYTHON_OP_LIB tf_python_op_lib_name)
    set(options SHAPE_FUNCTIONS_NOT_REQUIRED)
    set(oneValueArgs DESTINATION)
    set(multiValueArgs ADDITIONAL_LIBRARIES)
    cmake_parse_arguments(GENERATE_PYTHON_OP_LIB
      "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    if(NOT DEFINED GENERATE_PYTHON_OP_LIB_DESTINATION)
      # Default destination is tf_python/tensorflow/python/ops/gen_<...>.py.
      set(GENERATE_PYTHON_OP_LIB_DESTINATION
          "${python_ops_target_dir}/gen_${tf_python_op_lib_name}.py")
    endif()
    if(GENERATE_PYTHON_OP_LIB_SHAPE_FUNCTIONS_NOT_REQUIRED)
      set(require_shape_fn 0)
    else()
      set(require_shape_fn 1)
    endif()

    # Create a C++ executable that links in the appropriate op
    # registrations and generates Python wrapper code based on the
    # registered ops.
    add_executable(${tf_python_op_lib_name}_gen_python
        $<TARGET_OBJECTS:tf_python_op_gen_main>
        $<TARGET_OBJECTS:tf_${tf_python_op_lib_name}>
        ${GENERATE_PYTHON_OP_LIB_ADDITIONAL_LIBRARIES}
    )
    target_link_libraries(${tf_python_op_lib_name}_gen_python PRIVATE
        protobuf::libprotobuf
	tf_python_all_protos_cc
        tensorflow_core_lib
        tensorflow_protos
        ZLIB::ZLIB
        Snappy::snappy
        nsync
        dl
        ${tensorflow_EXTERNAL_LIBRARIES}
    )

    # Use the generated C++ executable to create a Python file
    # containing the wrappers.
    add_custom_command(
      OUTPUT ${GENERATE_PYTHON_OP_LIB_DESTINATION}
      COMMAND ${tf_python_op_lib_name}_gen_python @${tensorflow_source_dir}/tensorflow/python/ops/hidden_ops.txt ${require_shape_fn} > ${GENERATE_PYTHON_OP_LIB_DESTINATION}
      DEPENDS ${tf_python_op_lib_name}_gen_python
    )

    set(tf_python_ops_generated_files ${tf_python_ops_generated_files}
        ${GENERATE_PYTHON_OP_LIB_DESTINATION} PARENT_SCOPE)
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

set(tf_op_lib_names
    "array_ops"
    "bitwise_ops"
    "candidate_sampling_ops"
    "checkpoint_ops"
    "control_flow_ops"
    "ctc_ops"
    "data_flow_ops"
    "dataset_ops"
    "functional_ops"
    "image_ops"
    "io_ops"
    "linalg_ops"
    "lookup_ops"
    "logging_ops"
    "math_ops"
    "nn_ops"
    "no_op"
    "parsing_ops"
    "random_ops"
    "remote_fused_graph_ops"
    "resource_variable_ops"
    "script_ops"
    "sdca_ops"
    "set_ops"
    "sendrecv_ops"
    "sparse_ops"
    "spectral_ops"
    "state_ops"
    "stateless_random_ops"
    "string_ops"
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
    add_dependencies(tf_${tf_op_lib_name} tf_python_all_protos_cc)

endforeach()

file(GLOB tf_debug_ops_srcs
    "${tensorflow_source_dir}/tensorflow/core/ops/debug_ops.cc"
)

add_library(tf_debug_ops OBJECT ${tf_debug_ops_srcs})
add_dependencies(tf_debug_ops tf_python_protos_cc tf_python_all_protos_cc)

function(GENERATE_CONTRIB_OP_LIBRARY op_lib_name cc_srcs)
    add_library(tf_contrib_${op_lib_name}_ops OBJECT ${cc_srcs})
    add_dependencies(tf_contrib_${op_lib_name}_ops tf_python_all_protos_cc)
endfunction()

file(GLOB_RECURSE tensor_forest_hybrid_srcs
     "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/hybrid/core/ops/*.cc"
)

file(GLOB_RECURSE tpu_ops_srcs
     "${tensorflow_source_dir}/tensorflow/contrib/tpu/ops/*.cc"
)

GENERATE_CONTRIB_OP_LIBRARY(batch "${tensorflow_source_dir}/tensorflow/contrib/batching/ops/batch_ops.cc")
#GENERATE_CONTRIB_OP_LIBRARY(boosted_trees_ensemble_optimizer "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/ops/ensemble_optimizer_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(boosted_trees_model "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/ops/model_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(boosted_trees_split_handler "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/ops/split_handler_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(boosted_trees_training "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/ops/training_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(boosted_trees_prediction "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/ops/prediction_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(boosted_trees_quantiles "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/ops/quantile_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(boosted_trees_stats_accumulator "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/ops/stats_accumulator_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(cudnn_rnn "${tensorflow_source_dir}/tensorflow/contrib/cudnn_rnn/ops/cudnn_rnn_ops.cc")
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
GENERATE_CONTRIB_OP_LIBRARY(reduce_slice "${tensorflow_source_dir}/tensorflow/contrib/reduce_slice_ops/ops/reduce_slice_ops.cc")

GENERATE_PYTHON_OP_LIB("array_ops")
GENERATE_PYTHON_OP_LIB("bitwise_ops")
GENERATE_PYTHON_OP_LIB("math_ops")
GENERATE_PYTHON_OP_LIB("functional_ops")
GENERATE_PYTHON_OP_LIB("candidate_sampling_ops")
GENERATE_PYTHON_OP_LIB("checkpoint_ops")
GENERATE_PYTHON_OP_LIB("control_flow_ops"
  ADDITIONAL_LIBRARIES $<TARGET_OBJECTS:tf_no_op>)
GENERATE_PYTHON_OP_LIB("ctc_ops")
GENERATE_PYTHON_OP_LIB("data_flow_ops")
GENERATE_PYTHON_OP_LIB("dataset_ops")
GENERATE_PYTHON_OP_LIB("image_ops")
GENERATE_PYTHON_OP_LIB("io_ops")
GENERATE_PYTHON_OP_LIB("linalg_ops")
GENERATE_PYTHON_OP_LIB("logging_ops")
GENERATE_PYTHON_OP_LIB("lookup_ops")
GENERATE_PYTHON_OP_LIB("nn_ops")
GENERATE_PYTHON_OP_LIB("parsing_ops")
GENERATE_PYTHON_OP_LIB("random_ops")
GENERATE_PYTHON_OP_LIB("remote_fused_graph_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/remote_fused_graph/pylib/python/ops/gen_remote_fused_graph_ops.py)
GENERATE_PYTHON_OP_LIB("resource_variable_ops")
GENERATE_PYTHON_OP_LIB("script_ops")
GENERATE_PYTHON_OP_LIB("sdca_ops")
GENERATE_PYTHON_OP_LIB("set_ops")
GENERATE_PYTHON_OP_LIB("state_ops")
GENERATE_PYTHON_OP_LIB("sparse_ops")
GENERATE_PYTHON_OP_LIB("spectral_ops")
GENERATE_PYTHON_OP_LIB("string_ops")
GENERATE_PYTHON_OP_LIB("user_ops")
GENERATE_PYTHON_OP_LIB("training_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/python/training/gen_training_ops.py)

GENERATE_PYTHON_OP_LIB("contrib_batch_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/batching/ops/gen_batch_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_tpu_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/tpu/ops/gen_tpu_ops.py)
#GENERATE_PYTHON_OP_LIB("contrib_boosted_trees_ensemble_optimizer_ops"
#  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/boosted_trees/python/ops/gen_ensemble_optimizer_ops.py)

GENERATE_PYTHON_OP_LIB("contrib_boosted_trees_model_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/boosted_trees/python/ops/gen_model_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_boosted_trees_split_handler_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/boosted_trees/python/ops/gen_split_handler_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_boosted_trees_training_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/boosted_trees/python/ops/gen_training_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_boosted_trees_prediction_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/boosted_trees/python/ops/gen_prediction_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_boosted_trees_quantiles_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/boosted_trees/python/ops/gen_quantile_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_boosted_trees_stats_accumulator_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/boosted_trees/python/ops/gen_stats_accumulator_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_cudnn_rnn_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/cudnn_rnn/ops/gen_cudnn_rnn_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_factorization_clustering_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/factorization/python/ops/gen_clustering_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_factorization_factorization_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/factorization/python/ops/gen_factorization_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_framework_variable_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/framework/python/ops/gen_variable_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_input_pipeline_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/input_pipeline/ops/gen_input_pipeline_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_image_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/image/ops/gen_image_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_image_distort_image_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/image/ops/gen_distort_image_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_image_sirds_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/image/ops/gen_single_image_random_dot_stereograms_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_layers_sparse_feature_cross_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/layers/ops/gen_sparse_feature_cross_op.py)
GENERATE_PYTHON_OP_LIB("contrib_memory_stats_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/memory_stats/ops/gen_memory_stats_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_nccl_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/nccl/ops/gen_nccl_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_nearest_neighbor_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/nearest_neighbor/ops/gen_nearest_neighbor_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_resampler_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/resampler/ops/gen_resampler_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_rnn_gru_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/rnn/ops/gen_gru_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_rnn_lstm_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/rnn/ops/gen_lstm_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_seq2seq_beam_search_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/seq2seq/ops/gen_beam_search_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_tensor_forest_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/tensor_forest/python/ops/gen_tensor_forest_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_tensor_forest_hybrid_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/tensor_forest/hybrid/ops/gen_training_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_tensor_forest_model_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/tensor_forest/python/ops/gen_model_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_tensor_forest_stats_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/tensor_forest/python/ops/gen_stats_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_text_skip_gram_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/text/python/ops/gen_skip_gram_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_bigquery_reader_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/cloud/python/ops/gen_bigquery_reader_ops.py)
GENERATE_PYTHON_OP_LIB("stateless_random_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/stateless/gen_stateless_random_ops.py)
GENERATE_PYTHON_OP_LIB("debug_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/python/debug/ops/gen_debug_ops.py)

PYTHON_SYMLINK_CUSTOM_OPS_LIB(batch_ops "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/batch/python/ops/" "_batch_ops.so")
PYTHON_SYMLINK_CUSTOM_OPS_LIB(reduce_slice_ops "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/reduce_slice_ops/python/ops/" "_reduce_slice_ops.so")
PYTHON_SYMLINK_CUSTOM_OPS_LIB(skip_gram_ops "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/text/python/ops/" "_skip_gram_ops.so")
PYTHON_SYMLINK_CUSTOM_OPS_LIB(tpu_ops "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/tpu/python/ops/" "_tpu_ops.so")
PYTHON_SYMLINK_CUSTOM_OPS_LIB(boosted_trees_model_ops "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/boosted_trees/python/ops/" "_model_ops.so")
PYTHON_SYMLINK_CUSTOM_OPS_LIB(boosted_trees_split_handler_ops "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/boosted_trees/python/ops/" "_split_handler_ops.so")
PYTHON_SYMLINK_CUSTOM_OPS_LIB(boosted_trees_training_ops "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/boosted_trees/python/ops/" "_training_ops.so")
PYTHON_SYMLINK_CUSTOM_OPS_LIB(boosted_trees_quantile_ops "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/boosted_trees/python/ops/" "_quantile_ops.so")
PYTHON_SYMLINK_CUSTOM_OPS_LIB(boosted_trees_prediction_ops "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/boosted_trees/python/ops/" "_prediction_ops.so")
PYTHON_SYMLINK_CUSTOM_OPS_LIB(boosted_trees_stats_accumulator_ops "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/boosted_trees/python/ops/" "_stats_accumulator_ops.so")
PYTHON_SYMLINK_CUSTOM_OPS_LIB(variable_ops "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/framework/python/ops/" "_variable_ops.so")
PYTHON_SYMLINK_CUSTOM_OPS_LIB(memory_stats_ops "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/memory_stats/python/ops/" "_memory_stats_ops.so")
PYTHON_SYMLINK_CUSTOM_OPS_LIB(tensor_forest_model_ops "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/tensor_forest/python/ops/" "_model_ops.so")
PYTHON_SYMLINK_CUSTOM_OPS_LIB(tensor_forest_stats_ops "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/tensor_forest/python/ops/" "_stats_ops.so")
PYTHON_SYMLINK_CUSTOM_OPS_LIB(tensor_forest_ops "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/tensor_forest/python/ops/" "_tensor_forest_ops.so")
PYTHON_SYMLINK_CUSTOM_OPS_LIB(tensor_forest_training_ops "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/tensor_forest/hybrid/python/ops/" "_training_ops.so")
PYTHON_SYMLINK_CUSTOM_OPS_LIB(nearest_neighbor_ops "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/nearest_neighbor/python/ops/" "_nearest_neighbor_ops.so")
PYTHON_SYMLINK_CUSTOM_OPS_LIB(beam_search_ops "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/seq2seq/python/ops/" "_beam_search_ops.so")
PYTHON_SYMLINK_CUSTOM_OPS_LIB(cudnn_rnn_ops "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/cudnn_rnn/python/ops/" "_cudnn_rnn_ops.so")
PYTHON_SYMLINK_CUSTOM_OPS_LIB(resampler_ops "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/resampler/python/ops/" "_resampler_ops.so")
PYTHON_SYMLINK_CUSTOM_OPS_LIB(factorization_ops "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/factorization/python/ops/" "_factorization_ops.so")
PYTHON_SYMLINK_CUSTOM_OPS_LIB(clustering_ops "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/factorization/python/ops/" "_clustering_ops.so")
PYTHON_SYMLINK_CUSTOM_OPS_LIB(input_pipeline_ops "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/input_pipeline/python/ops/" "_input_pipeline_ops.so")
PYTHON_SYMLINK_CUSTOM_OPS_LIB(image_ops "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/image/python/ops/" "_image_ops.so")
PYTHON_SYMLINK_CUSTOM_OPS_LIB(single_image_random_dot_stereograms "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/image/python/ops/" "_single_image_random_dot_stereograms.so")
PYTHON_SYMLINK_CUSTOM_OPS_LIB(distort_image_ops "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/image/python/ops/" "_distort_image_ops.so")
PYTHON_SYMLINK_CUSTOM_OPS_LIB(nccl_ops "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/nccl/python/ops/" "_nccl_ops.so")
PYTHON_SYMLINK_CUSTOM_OPS_LIB(sparse_feature_cross_op "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/layers/python/ops/" "_sparse_feature_cross_op.so")
PYTHON_SYMLINK_CUSTOM_OPS_LIB(gru_ops "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/rnn/python/ops/" "_gru_ops.so")
PYTHON_SYMLINK_CUSTOM_OPS_LIB(lstm_ops "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/rnn/python/ops/" "_lstm_ops.so")

add_custom_target(tf_python_ops SOURCES ${tf_python_ops_generated_files} ${PYTHON_PROTO_GENFILES})
add_dependencies(tf_python_ops tf_python_op_gen_main tf_python_all_protos_cc)

find_package(SWIG REQUIRED)
# Generate the C++ and Python source code for the SWIG wrapper.
# NOTE(mrry): We always regenerate the SWIG wrapper, which means that we must
# always re-link the Python extension, but we don't have to track the
# individual headers on which the SWIG wrapper depends.
add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/python/pywrap_tensorflow_internal.py"
             "${CMAKE_CURRENT_BINARY_DIR}/pywrap_tensorflow_internal.cc"
      DEPENDS tf_python_touchup_modules
      COMMAND ${SWIG_EXECUTABLE}
      ARGS -python -c++
           -I${tensorflow_source_dir}
           -I${CMAKE_CURRENT_BINARY_DIR}
           -module pywrap_tensorflow_internal
           -outdir ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/python
           -o ${CMAKE_CURRENT_BINARY_DIR}/pywrap_tensorflow_internal.cc
           -globals ''
           ${tensorflow_source_dir}/tensorflow/python/tensorflow.i
      COMMENT "Running SWIG to generate Python wrappers"
      VERBATIM )

set (pywrap_tensorflow_internal_src
    "${tensorflow_source_dir}/tensorflow/core/profiler/internal/print_model_analysis.h"
    "${tensorflow_source_dir}/tensorflow/core/profiler/internal/print_model_analysis.cc"
    "${tensorflow_source_dir}/tensorflow/python/eager/pywrap_tfe.h"
    "${tensorflow_source_dir}/tensorflow/python/eager/pywrap_tensor.cc"
    "${tensorflow_source_dir}/tensorflow/python/eager/pywrap_tfe_src.cc"
    "${tensorflow_source_dir}/tensorflow/python/client/tf_session_helper.h"
    "${tensorflow_source_dir}/tensorflow/python/client/tf_session_helper.cc"
    "${tensorflow_source_dir}/tensorflow/python/eager/python_eager_op_gen.h"
    "${tensorflow_source_dir}/tensorflow/python/eager/python_eager_op_gen.cc"
    "${tensorflow_source_dir}/tensorflow/python/framework/cpp_shape_inference.h"
    "${tensorflow_source_dir}/tensorflow/python/framework/cpp_shape_inference.cc"
    "${tensorflow_source_dir}/tensorflow/python/framework/python_op_gen.h"
    "${tensorflow_source_dir}/tensorflow/python/framework/python_op_gen.cc"
    "${tensorflow_source_dir}/tensorflow/python/lib/core/numpy.h"
    "${tensorflow_source_dir}/tensorflow/python/lib/core/numpy.cc"
    "${tensorflow_source_dir}/tensorflow/python/lib/core/ndarray_tensor.h"
    "${tensorflow_source_dir}/tensorflow/python/lib/core/ndarray_tensor.cc"
    "${tensorflow_source_dir}/tensorflow/python/lib/core/ndarray_tensor_bridge.h"
    "${tensorflow_source_dir}/tensorflow/python/lib/core/ndarray_tensor_bridge.cc"
    "${tensorflow_source_dir}/tensorflow/python/lib/core/py_func.h"
    "${tensorflow_source_dir}/tensorflow/python/lib/core/py_func.cc"
    "${tensorflow_source_dir}/tensorflow/python/lib/core/py_seq_tensor.h"
    "${tensorflow_source_dir}/tensorflow/python/lib/core/py_seq_tensor.cc"
    "${tensorflow_source_dir}/tensorflow/python/lib/core/safe_ptr.h"
    "${tensorflow_source_dir}/tensorflow/python/lib/core/safe_ptr.cc"
    "${tensorflow_source_dir}/tensorflow/python/lib/io/py_record_reader.h"
    "${tensorflow_source_dir}/tensorflow/python/lib/io/py_record_reader.cc"
    "${tensorflow_source_dir}/tensorflow/python/lib/io/py_record_writer.h"
    "${tensorflow_source_dir}/tensorflow/python/lib/io/py_record_writer.cc"
    "${tensorflow_source_dir}/tensorflow/python/util/kernel_registry.h"
    "${tensorflow_source_dir}/tensorflow/python/util/kernel_registry.cc"
    "${tensorflow_source_dir}/tensorflow/cc/framework/ops.cc"
    "${tensorflow_source_dir}/tensorflow/cc/framework/scope.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/pywrap_tensorflow_internal.cc"
)

# pywrap_tensorflow_internal is a shared library containing all of the
# TensorFlow runtime and the standard ops and kernels. These are installed into
# tf_python/tensorflow/python/.
add_library(pywrap_tensorflow_internal SHARED
    ${pywrap_tensorflow_internal_src}
    ${pywrap_tensorflow_deffile}
)

if(WIN32)
    add_dependencies(pywrap_tensorflow_internal pywrap_tensorflow_internal_static)
endif(WIN32)

target_include_directories(pywrap_tensorflow_internal PUBLIC
    ${PYTHON_INCLUDE_DIR}
    ${NUMPY_INCLUDE_DIR}
)

target_link_libraries(pywrap_tensorflow_internal PRIVATE
    ${tf_core_gpu_kernels_lib}
    tf_python_protos_cc
    ${PYTHON_LIBRARIES}
    tensorflow
    tensorflow_grappler
    jsoncpp_lib
)
if(tensorflow_ENABLE_MPI)
    target_link_libraries(pywrap_tensorflow_internal PRIVATE
        tensorflow_mpi)
endif()
add_dependencies(pywrap_tensorflow_internal tf_python_ops)

add_custom_target(tf_python_build_pip_package ALL)
add_dependencies(tf_python_build_pip_package
    pywrap_tensorflow_internal
    python_library_symlinks
    tf_python_copy_scripts_to_destination
    tf_python_touchup_modules
    tf_python_ops)
add_dependencies(tf_python_copy_scripts_to_destination
    tf_python_ops
    tf_python_touchup_modules
    python_library_symlinks
)


# Fix-up Python files that were not included by the add_python_module() macros.
add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${tensorflow_source_dir}/tensorflow/tools/pip_package/setup.py
                                   ${CMAKE_CURRENT_BINARY_DIR}/tf_python/)
# This file is unfortunately excluded by the regex that excludes *_test.py
# files, but it is imported into tf.contrib, so we add it explicitly.
add_custom_command(TARGET tf_python_copy_scripts_to_destination PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${tensorflow_source_dir}/tensorflow/contrib/testing/python/framework/util_test.py
                                   ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/testing/python/framework/)

if(WIN32)
  add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$(Configuration)/pywrap_tensorflow_internal.dll
                                     ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/python/_pywrap_tensorflow_internal.pyd
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$(Configuration)/pywrap_tensorflow_internal.lib
                                     ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/python/)
else()
  add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/libpywrap_tensorflow_internal.so
                                     ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/python/_pywrap_tensorflow_internal.so)
#  add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/libtensor_forest_protos.so
#                                     ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/libtensor_forest_protos.so)
endif()
add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${tensorflow_source_dir}/tensorflow/tools/pip_package/README
                                   ${CMAKE_CURRENT_BINARY_DIR}/tf_python/)
add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${tensorflow_source_dir}/tensorflow/tools/pip_package/MANIFEST.in
                                   ${CMAKE_CURRENT_BINARY_DIR}/tf_python/)

# Copy datasets for tf.contrib.learn.
add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${tensorflow_source_dir}/tensorflow/contrib/learn/python/learn/datasets/data/boston_house_prices.csv
                                   ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/learn/python/learn/datasets/data/)
add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${tensorflow_source_dir}/tensorflow/contrib/learn/python/learn/datasets/data/iris.csv
                                   ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/learn/python/learn/datasets/data/)
add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${tensorflow_source_dir}/tensorflow/contrib/learn/python/learn/datasets/data/text_test.csv
                                   ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/learn/python/learn/datasets/data/)
add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${tensorflow_source_dir}/tensorflow/contrib/learn/python/learn/datasets/data/text_train.csv
                                   ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/learn/python/learn/datasets/data/)

# Create include header directory
add_custom_command(TARGET tf_python_build_pip_package PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
  ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/include/)

# tensorflow headers
add_custom_command(TARGET tf_python_build_pip_package PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
  ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/include/tensorflow)
add_custom_command(TARGET tf_python_build_pip_package PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
  ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/include/tensorflow/core)
add_custom_command(TARGET tf_python_build_pip_package PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
  ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/include/tensorflow/stream_executor)
add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${tensorflow_source_dir}/tensorflow/core
                                   ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/include/tensorflow/core)
add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/core
                                   ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/include/tensorflow/core)
add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${tensorflow_source_dir}/tensorflow/stream_executor
                                   ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/include/tensorflow/stream_executor)
