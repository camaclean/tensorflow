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
if(WIN32)
  # Windows: build a static library with the same objects as tensorflow.dll.
  # This can be used to build for a standalone exe and also helps us to
  # find all symbols that need to be exported from the dll which is needed
  # to provide the tensorflow c/c++ api in tensorflow.dll.
  # From the static library we create the def file with all symbols that need to
  # be exported from tensorflow.dll. Because there is a limit of 64K sybmols
  # that can be exported, we filter the symbols with a python script to the namespaces
  # we need.
  #
  add_library(tensorflow_static STATIC
      $<TARGET_OBJECTS:tf_c>
      $<TARGET_OBJECTS:tf_cc_framework>
      $<TARGET_OBJECTS:tf_cc_ops>
      $<TARGET_OBJECTS:tf_cc_while_loop>
      $<TARGET_OBJECTS:tf_core_lib>
      $<TARGET_OBJECTS:tf_core_cpu>
      $<TARGET_OBJECTS:tf_core_framework>
      $<TARGET_OBJECTS:tf_core_ops>
      $<TARGET_OBJECTS:tf_core_direct_session>
      $<TARGET_OBJECTS:tf_tools_transform_graph_lib>
      $<$<BOOL:${tensorflow_ENABLE_GRPC_SUPPORT}>:$<TARGET_OBJECTS:tf_core_distributed_runtime>>
      $<TARGET_OBJECTS:tf_core_kernels>
      $<$<BOOL:${tensorflow_ENABLE_GPU}>:$<TARGET_OBJECTS:tf_core_kernels_cpu_only>>
      $<$<BOOL:${tensorflow_ENABLE_GPU}>:$<TARGET_OBJECTS:tf_stream_executor>>
  )

  add_dependencies(tensorflow_static tensorflow_protos)
  set(tensorflow_static_dependencies
      $<TARGET_FILE:tensorflow_static>
      $<TARGET_FILE:tensorflow_protos>
  )

  if(${CMAKE_GENERATOR} MATCHES "Visual Studio.*")
    set(tensorflow_deffile "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/tensorflow.def")
  else()
    set(tensorflow_deffile "${CMAKE_CURRENT_BINARY_DIR}/tensorflow.def")
  endif()
  set_source_files_properties(${tensorflow_deffile} PROPERTIES GENERATED TRUE)
  math(EXPR tensorflow_target_bitness "${CMAKE_SIZEOF_VOID_P}*8")
  add_custom_command(TARGET tensorflow_static POST_BUILD
      COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/create_def_file.py
          --input "${tensorflow_static_dependencies}"
          --output "${tensorflow_deffile}"
          --target tensorflow.dll
          --bitness "${tensorflow_target_bitness}"
  )
endif(WIN32)

#        "//tensorflow/core:framework_internal_impl",
#        "//tensorflow/core:lib_internal_impl",
#        "//tensorflow/core:core_cpu_impl",
#        "//tensorflow/stream_executor:stream_executor_impl",
#        "//tensorflow/core:gpu_runtime_impl",

add_library(tensorflow_core_lib STATIC
        $<TARGET_OBJECTS:tf_core_lib>
        $<TARGET_OBJECTS:tf_core_framework>
)
add_dependencies(tensorflow_core_lib tf_core_lib tf_core_framework)
list(APPEND tensorflow_libs tensorflow_core_lib)
#list(APPEND tensorflow_libs tensorflow_protos)
add_library(tensorflow_framework SHARED
    $<$<BOOL:${tensorflow_ENABLE_GPU}>:$<TARGET_OBJECTS:tf_stream_executor>>
    $<TARGET_OBJECTS:tf_c>
#    $<TARGET_OBJECTS:tf_c_python_api> #may depend on version specfic stuff
    $<TARGET_OBJECTS:tf_core_lib>
    $<TARGET_OBJECTS:tf_core_cpu>
    $<TARGET_OBJECTS:tf_cc_framework>
    $<TARGET_OBJECTS:tf_core_framework>
)
set_target_properties(tensorflow_framework PROPERTIES
    VERSION ${TENSORFLOW_LIB_VERSION}
    SOVERSION ${TENSORFLOW_LIB_SOVERSION}
)
target_link_libraries(tensorflow_framework PRIVATE
    ${tf_core_gpu_kernels_lib}
    ${tensorflow_EXTERNAL_LIBRARIES}
    tensorflow_text_protos
    tensorflow_protos)
list(APPEND tensorflow_libs tensorflow_framework)

if(WIN32)
  add_library(tensorflow SHARED
      $<TARGET_OBJECTS:tf_cc>
      $<TARGET_OBJECTS:tf_cc_ops>
      $<TARGET_OBJECTS:tf_cc_while_loop>
      $<TARGET_OBJECTS:tf_core_ops>
      $<TARGET_OBJECTS:tf_core_kernels>
      $<TARGET_OBJECTS:tf_core_profiler>
      $<TARGET_OBJECTS:tf_core_direct_session>
      $<TARGET_OBJECTS:tf_tools_transform_graph_lib>
      $<$<BOOL:${tensorflow_ENABLE_GRPC_SUPPORT}>:$<TARGET_OBJECTS:tf_core_distributed_runtime>>
      $<$<BOOL:${tensorflow_ENABLE_GPU}>:$<TARGET_OBJECTS:tf_core_kernels_cpu_only>>
      ${tensorflow_deffile}
  )
else()
  add_library(tensorflow SHARED
      $<TARGET_OBJECTS:tf_cc>
      $<TARGET_OBJECTS:tf_cc_ops>
      $<TARGET_OBJECTS:tf_cc_while_loop>
      $<TARGET_OBJECTS:tf_core_ops>
      $<TARGET_OBJECTS:tf_core_kernels>
      $<TARGET_OBJECTS:tf_core_profiler>
      $<TARGET_OBJECTS:tf_core_direct_session>
      $<TARGET_OBJECTS:tf_tools_transform_graph_lib>
      $<$<BOOL:${tensorflow_ENABLE_GRPC_SUPPORT}>:$<TARGET_OBJECTS:tf_core_distributed_runtime>>
      ${tensorflow_deffile}
  )
endif()
set_target_properties(tensorflow PROPERTIES
    VERSION ${TENSORFLOW_LIB_VERSION}
    SOVERSION ${TENSORFLOW_LIB_SOVERSION})

target_link_libraries(tensorflow PUBLIC tensorflow_framework tensorflow_text_protos)
target_link_libraries(tensorflow PRIVATE
    ${tf_core_gpu_kernels_lib}
    ${tensorflow_EXTERNAL_LIBRARIES}
    #tensorflow_protos
)
target_include_directories(tensorflow PUBLIC 
  $<BUILD_INTERFACE:${tensorflow_source_dir}/tensorflow/c>
  $<INSTALL_INTERFACE:include/tensorflow/c>  # <prefix>/include/tensorflow/c
)
set_target_properties(tensorflow PROPERTIES PUBLIC_HEADER 
  "${tensorflow_source_dir}/tensorflow/c/python_api.h;${tensorflow_source_dir}/tensorflow/c/c_api.h")
install(TARGETS tensorflow
    EXPORT TensorflowTargets
    LIBRARY DESTINATION "lib${LIBSUFFIX}"
    ARCHIVE DESTINATION "lib${LIBSUFFIX}"
    INCLUDES DESTINATION include/tensorflow/c
    PUBLIC_HEADER DESTINATION "include/tensorflow/c")

install(TARGETS ${tensorflow_libs}
    EXPORT TensorflowTargets
    LIBRARY DESTINATION "lib${LIBSUFFIX}"
    ARCHIVE DESTINATION "lib${LIBSUFFIX}"
    PUBLIC_HEADER DESTINATION "include/tensorflow/c")

# Create TensorflowConfig.cmake
EXPORT(TARGETS ${tensorflow_libs} tensorflow_protos tensorflow_text_protos FILE "${CMAKE_CURRENT_BINARY_DIR}/TensorflowTargets.cmake")
INSTALL(EXPORT TensorflowTargets DESTINATION "share/tensorflow/cmake")

# There is a bug in GCC 5 resulting in undefined reference to a __cpu_model function when
# linking to the tensorflow library. Adding the following libraries fixes it.
# See issue on github: https://github.com/tensorflow/tensorflow/issues/9593
if(CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0)
    target_link_libraries(tensorflow PRIVATE gcc_s gcc)
endif()

if(WIN32)
  add_dependencies(tensorflow tensorflow_static)
endif(WIN32)

target_include_directories(tensorflow PUBLIC 
    $<INSTALL_INTERFACE:include/>)

install(TARGETS tensorflow EXPORT tensorflow_export
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
        
install(EXPORT tensorflow_export
        FILE TensorflowConfig.cmake
        DESTINATION lib/cmake)

# install necessary headers
# tensorflow headers
install(DIRECTORY ${tensorflow_source_dir}/tensorflow/cc/
        DESTINATION include/tensorflow/cc
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/cc/
        DESTINATION include/tensorflow/cc
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ${tensorflow_source_dir}/tensorflow/core/
        DESTINATION include/tensorflow/core
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/core/
        DESTINATION include/tensorflow/core
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ${tensorflow_source_dir}/tensorflow/stream_executor/
        DESTINATION include/tensorflow/stream_executor
        FILES_MATCHING PATTERN "*.h")
