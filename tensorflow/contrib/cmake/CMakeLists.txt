# Minimum CMake required
cmake_minimum_required(VERSION 3.5)

# Project
project(tensorflow C CXX)

# Set C++14 as standard for the whole project
set(CMAKE_CXX_STANDARD 14)

# Actual source is the ../../.. directory
get_filename_component(tf_contrib_source_dir ${tensorflow_SOURCE_DIR} PATH)
get_filename_component(tf_tf_source_dir ${tf_contrib_source_dir} PATH)
get_filename_component(tensorflow_source_dir ${tf_tf_source_dir} PATH)

# [CLEANUP] Not sure if this is needed (copied from Protobuf)
# CMake policies
cmake_policy(SET CMP0022 NEW)

get_property(LIB64 GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS)

if ("${LIB64}" STREQUAL "TRUE")
    set(LIBSUFFIX 64)
else()
    set(LIBSUFFIX "")
endif()

set(TENSORFLOW_LIB_VERSION 1.4.1)
set(TENSORFLOW_LIB_SOVERSION 1)

# Options
option(tensorflow_VERBOSE "Enable for verbose output" OFF)
option(tensorflow_ENABLE_GPU "Enable GPU support" OFF)
option(tensorflow_CUDA_30 "Build for CUDA 3.0 compute capability" ON)
option(tensorflow_CUDA_35 "Build for CUDA 3.5 compute capability" ON)
option(tensorflow_CUDA_52 "Build for CUDA 5.2 compute capability" ON)
option(tensorflow_ENABLE_SSL_SUPPORT "Enable boringssl support" OFF)
option(tensorflow_ENABLE_GRPC_SUPPORT "Enable gRPC support" ON)
option(tensorflow_ENABLE_HDFS_SUPPORT "Enable HDFS support" OFF)
option(tensorflow_ENABLE_JEMALLOC_SUPPORT "Enable jemalloc support" OFF)
option(tensorflow_BUILD_CC_EXAMPLE "Build the C++ tutorial example" ON)
option(tensorflow_BUILD_PYTHON_BINDINGS "Build the Python bindings" ON)
option(tensorflow_BUILD_ALL_KERNELS "Build all OpKernels" ON)
option(tensorflow_BUILD_CONTRIB_KERNELS "Build OpKernels from tensorflow/contrib/..." ON)
option(tensorflow_BUILD_CC_TESTS "Build cc unit tests " OFF)
option(tensorflow_BUILD_PYTHON_TESTS "Build python unit tests " OFF)
option(tensorflow_BUILD_MORE_PYTHON_TESTS "Build more python unit tests for contrib packages" OFF)
option(tensorflow_BUILD_SHARED_LIB "Build TensorFlow as a shared library" OFF)
option(tensorflow_OPTIMIZE_FOR_NATIVE_ARCH "Enable compiler optimizations for the native processor architecture (if available)" ON)
option(tensorflow_WIN_CPU_SIMD_OPTIONS "Enables CPU SIMD instructions")
option(tensorflow_ENABLE_SNAPPY_SUPPORT "Enable SNAPPY compression support" ON)
option(tensorflow_ENABLE_MPI "Enable MPI support" OFF)
option(tensorflow_USE_MKL OFF)
option(LIBTENSORFLOW_MODULE ON)

if (LIBTENSORFLOW_MODULE)
    set(tensorflow_BUILD_PYTHON_BINDINGS OFF)
    set(tensorflow_BUILD_SHARED_LIB "Build TensorFlow as a shared library" ON)
endif()

if(tensorflow_USE_MKL)
    add_definitions(-DINTEL_MKL=1 -fopenmp)
    include_directories("${MKL_INCLUDE_DIR}")
endif()

if(UNIX)
    set(CMAKE_EXE_LINKER_FLAGS "$ENV{LDFLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "$ENV{LDFLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

if (NOT WIN32)
  # Threads: defines CMAKE_THREAD_LIBS_INIT and adds -pthread compile option
  # for targets that link ${CMAKE_THREAD_LIBS_INIT}.
  find_package (Threads)
endif()

# [CLEANUP] Remove when done
# For debugging
function(SHOW_VARIABLES)
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
endfunction()

# External dependencies
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/external)

# Location where external projects will be downloaded
set (DOWNLOAD_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/downloads"
     CACHE PATH "Location where external projects will be downloaded.")
mark_as_advanced(DOWNLOAD_LOCATION)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_definitions(-DEIGEN_AVOID_STL_ARRAY)
if(WIN32)
  add_definitions(-DNOMINMAX -D_WIN32_WINNT=0x0A00 -DLANG_CXX11 -DCOMPILER_MSVC)
  add_definitions(-DWIN32 -DOS_WIN -D_MBCS -DWIN64 -DWIN32_LEAN_AND_MEAN -DNOGDI -DPLATFORM_WINDOWS)
  add_definitions(-DTENSORFLOW_USE_EIGEN_THREADPOOL -DEIGEN_HAS_C99_MATH)
  add_definitions(-DTF_COMPILE_LIBRARY)
  add_definitions(/bigobj /nologo /EHsc /GF /MP /Gm-)
  # Suppress warnings to reduce build log size.
  add_definitions(/wd4267 /wd4244 /wd4800 /wd4503 /wd4554 /wd4996 /wd4348 /wd4018)
  add_definitions(/wd4099 /wd4146 /wd4267 /wd4305 /wd4307)
  add_definitions(/wd4715 /wd4722 /wd4723 /wd4838 /wd4309 /wd4334)
  add_definitions(/wd4003 /wd4244 /wd4267 /wd4503 /wd4506 /wd4800 /wd4996)
  # Suppress linker warnings.
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /ignore:4049 /ignore:4197 /ignore:4217 /ignore:4221")
  set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /ignore:4049 /ignore:4197 /ignore:4217 /ignore:4221")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ignore:4049 /ignore:4197 /ignore:4217 /ignore:4221")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
  set(CMAKE_CXX_FLAGS_DEBUG "/D_DEBUG /MDd /Ob2")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /D_ITERATOR_DEBUG_LEVEL=0")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /D_ITERATOR_DEBUG_LEVEL=0")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /D_ITERATOR_DEBUG_LEVEL=0")
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -std=c++11")
endif()

if (tensorflow_OPTIMIZE_FOR_NATIVE_ARCH)
  include(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
  if (COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
  endif()
endif()

# MSVC SIMD instructions
if (tensorflow_WIN_CPU_SIMD_OPTIONS)
  if (WIN32)
    CHECK_CXX_COMPILER_FLAG("${tensorflow_WIN_CPU_SIMD_OPTIONS}" COMPILER_OPT_WIN_CPU_SIMD_SUPPORTED)
    if(COMPILER_OPT_WIN_CPU_SIMD_SUPPORTED)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${tensorflow_WIN_CPU_SIMD_OPTIONS}")
    else()
      message(FATAL_ERROR "${tensorflow_WIN_CPU_SIMD_OPTIONS} not supported")
    endif()
  endif()
endif()

if (tensorflow_ENABLE_JEMALLOC_SUPPORT)
  add_definitions(-DTENSORFLOW_USE_JEMALLOC -DJEMALLOC_EXPORT=)
endif()

# External dependencies
include(FindPkgConfig)
find_package(ZLIB REQUIRED)
find_package(GIF REQUIRED)
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)
find_package(LMDB REQUIRED)
find_package(Eigen3Tensorflow REQUIRED CONFIG)
#find_package(jsoncpp)
include(gemmlowp)
include(jsoncpp)
include(farmhash)
include(fft2d)
find_package(highwayhash REQUIRED)
find_package(nsync REQUIRED)
set(protobuf_MODULE_COMPATIBLE ON)
find_package(protobuf REQUIRED)
#include(re2)
pkg_check_modules(RE2 REQUIRED re2)
include(cub)
include(sqlite) #done
if (tensorflow_BUILD_CC_TESTS)
  include(googletest)
endif()

set(tensorflow_EXTERNAL_LIBRARIES
    ZLIB::ZLIB
    ${GIF_LIBRARIES}
    PNG::PNG
    ${JPEG_LIBRARIES}
    LMDB::LMDB
    Eigen3::Eigen
    nsync
    highwayhash
    ${jsoncpp_STATIC_LIBRARIES}
    #jsoncpp_lib
    ${farmhash_STATIC_LIBRARIES}
    ${fft2d_STATIC_LIBRARIES}
    ${nsync_STATIC_LIBRARIES}
    protobuf::libprotobuf
    ${RE2_LIBRARIES}
    ${sqlite_STATIC_LIBRARIES}
    ${MKL_LIBRARIES}
)
set(tensorflow_EXTERNAL_DEPENDENCIES
    farmhash_copy_headers_to_destination
    gemmlowp
    cub
    fft2d
)
#get_target_property(JSONCPP_INCLUDE_DIR jsoncpp_lib
#  INTERFACE_INCLUDE_DIRECTORIES
#)

include_directories(
    # Source and generated code.
    ${tensorflow_source_dir}
    ${CMAKE_CURRENT_BINARY_DIR}
    # External dependencies.
    #${ZLIB_INCLUDE_DIRS}
    ${GIF_INCLUDE_DIR}
    ${JPEG_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIRS}
    ${gemmlowp_INCLUDE_DIR}
    "${jsoncpp_INCLUDE_DIR}/include"
    ${farmhash_INCLUDE_DIR}
    ${HIGHWAYHASH_INCLUDE_DIR}
    ${cub_INCLUDE_DIR}
    ${NSYNC_INCLUDE_DIR}
    #${PROTOBUF_INCLUDE_DIRS}
    #${re2_INCLUDE_DIR}
    ${RE2_INCLUDE_DIRS}
    ${sqlite_INCLUDE_DIR}
)

if(tensorflow_ENABLE_SSL_SUPPORT)
  include(boringssl)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES ${boringssl_STATIC_LIBRARIES})
  list(APPEND tensorflow_EXTERNAL_DEPENDENCIES boringssl)
  include_directories(${boringssl_INCLUDE_DIR})
endif()
if(tensorflow_ENABLE_GRPC_SUPPORT)
  include(grpc)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES ${grpc_STATIC_LIBRARIES})
endif()
if(tensorflow_ENABLE_JEMALLOC_SUPPORT)
  include(jemalloc)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES ${jemalloc_STATIC_LIBRARIES})
  list(APPEND tensorflow_EXTERNAL_DEPENDENCIES jemalloc)
  include_directories(${jemalloc_INCLUDE_DIRS})
endif()
if(tensorflow_ENABLE_SNAPPY_SUPPORT)
  find_package(Snappy REQUIRED)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES Snappy::snappy)
endif()
if(WIN32)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES wsock32 ws2_32 shlwapi)
endif()
if(UNIX)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})
endif()

function(JOIN VALUES GLUE OUTPUT)
  string (REGEX REPLACE "([^\\]|^);" "\\1${GLUE}" _TMP_STR "${VALUES}")
  string (REGEX REPLACE "[\\](.)" "\\1" _TMP_STR "${_TMP_STR}") #fixes escaping
  set (${OUTPUT} "${_TMP_STR}" PARENT_SCOPE)
endfunction()

if(NOT (tensorflow_CUDA_30 OR tensorflow_CUDA_35 OR tensorflow_CUDA_52))
    set(tensorflow_ENABLE_GPU FALSE)
endif()

if (tensorflow_ENABLE_GPU)
  find_package(CUDA 7.5 REQUIRED)
  set(CUDA_TOOLKIT_TARGET_DIR "${CUDA_TOOLKIT_ROOT_DIR}" CACHE STRING "")

  # by default we assume compute cabability 3.5 and 5.2. If you change this change it in
  # CUDA_NVCC_FLAGS and cuda_config.h below
  set(TF_EXTRA_CUDA_CAPABILITIES_30 "3.0")
  set(TF_EXTRA_CUDA_CAPABILITIES_35 "3.5")
  set(TF_EXTRA_CUDA_CAPABILITIES_52 "5.2")
  set(CUDA_NVCC_FLAGS_30 "-gencode arch=compute_30,code=\"sm_30,compute_30\"")
  set(CUDA_NVCC_FLAGS_35 "-gencode arch=compute_35,code=\"sm_35,compute_35\"")
  set(CUDA_NVCC_FLAGS_52 "-gencode arch=compute_52,code=\"sm_30,compute_52\"")
  set(TF_CUDA_CAPABILITIES_30 "CudaVersion(\"3.0\")")
  set(TF_CUDA_CAPABILITIES_35 "CudaVersion(\"3.5\")")
  set(TF_CUDA_CAPABILITIES_52 "CudaVersion(\"5.2\")")
  if(tensorflow_CUDA_30)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${CUDA_NVCC_FLAGS_30})
    set(TF_CUDA_CAPABILITIES_LIST ${TF_CUDA_CAPABILITIES_LIST} ${TF_CUDA_CAPABILITIES_30})
    set(TF_EXTRA_CUDA_CAPABILITIES_LIST ${TF_EXTRA_CUDA_CAPABILITIES_LIST} ${TF_EXTRA_CUDA_CAPABILITIES_30})
  endif()
  if(tensorflow_CUDA_35)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${CUDA_NVCC_FLAGS_35})
    set(TF_CUDA_CAPABILITIES_LIST ${TF_CUDA_CAPABILITIES_LIST} ${TF_CUDA_CAPABILITIES_35})
    set(TF_EXTRA_CUDA_CAPABILITIES_LIST ${TF_EXTRA_CUDA_CAPABILITIES_LIST} ${TF_EXTRA_CUDA_CAPABILITIES_35})
  endif()
  if(tensorflow_CUDA_52)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${CUDA_NVCC_FLAGS_52})
    set(TF_CUDA_CAPABILITIES_LIST ${TF_CUDA_CAPABILITIES_LIST} ${TF_CUDA_CAPABILITIES_52})
    set(TF_EXTRA_CUDA_CAPABILITIES_LIST ${TF_EXTRA_CUDA_CAPABILITIES_LIST} ${TF_EXTRA_CUDA_CAPABILITIES_52})
  endif()
  join("${TF_CUDA_CAPABILITIES_LIST}" "," TF_CUDA_CAPABILITIES)
  join("${TF_EXTRA_CUDA_CAPABILITIES_LIST}" "," TF_EXTRA_CUDA_CAPABILITIES)
  
  if(${CUDA_VERSION} STREQUAL "7.5")
    if(WIN32)
      set(TF_CUDA_VERSION "64_75")
    else()
      set(TF_CUDA_VERSION "7.5")
    endif()
  elseif(${CUDA_VERSION} STREQUAL "8.0")
    if(WIN32)
      set(TF_CUDA_VERSION "64_80")
    else()
      set(TF_CUDA_VERSION "8.0")
    endif()
  elseif(${CUDA_VERSION} STREQUAL "9.0")
    if(WIN32)
      set(TF_CUDA_VERSION "64_90")
    else()
      set(TF_CUDA_VERSION "9.0")
    endif()
  elseif(${CUDA_VERSION} STREQUAL "9.1")
    if(WIN32)
      set(TF_CUDA_VERSION "64_91")
    else()
      set(TF_CUDA_VERSION "9.1")
    endif()
  elseif(${CUDA_VERSION} STREQUAL "9.2")
    if(WIN32)
      set(TF_CUDA_VERSION "64_92")
    else()
      set(TF_CUDA_VERSION "9.2")
    endif()
  endif()
  if(NOT CUDNN_VERSION)
    set(CUDNN_VERSION "5")
  endif(NOT CUDNN_VERSION)
  if(WIN32)
    set(TF_CUDNN_VERSION "64_${CUDNN_VERSION}")
  else()
    set(TF_CUDNN_VERSION "${CUDNN_VERSION}")
  endif()

  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--include-path ${PROJECT_BINARY_DIR}/$\{build_configuration\};--expt-relaxed-constexpr)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-ftz=true)  # Flush denormals to zero
  if(UNIX)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--rpath=${CUDA_TOOLKIT_TARGET_DIR}/lib64 -Wl,--rpath=${CUDA_TOOLKIT_TARGET_DIR}/extras/CUPTI/lib64")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--rpath=${CUDA_TOOLKIT_TARGET_DIR}/lib64 -Wl,--rpath=${CUDA_TOOLKIT_TARGET_DIR}/extras/CUPTI/lib64")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xcompiler;-fPIC)
  endif(UNIX)
  set(CUDA_INCLUDE ${CUDA_TOOLKIT_TARGET_DIR}/include ${CUDA_TOOLKIT_TARGET_DIR}/include/crt ${CUDA_TOOLKIT_TARGET_DIR}/extras/CUPTI/include)
  include_directories(${CUDA_INCLUDE})
  add_definitions(-DGOOGLE_CUDA=1 -DTF_EXTRA_CUDA_CAPABILITIES=${TF_EXTRA_CUDA_CAPABILITIES})
  if (WIN32)
    # add cudnn
    if(NOT CUDNN_HOME)
      set(CUDNN_HOME ${CUDA_TOOLKIT_TARGET_DIR})
    endif(NOT CUDNN_HOME)
    include_directories(${CUDNN_HOME})
    set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUFFT_LIBRARIES}
      ${CUDA_curand_LIBRARY} ${CUDA_cupti_LIBRARY} ${CUDA_cusolver_LIBRARY} ${CUDNN_HOME}/lib/x64/cudnn.lib)
  elseif(UNIX)
    set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUFFT_LIBRARIES}
      ${CUDA_curand_LIBRARY} ${CUDA_cupti_LIBRARY} ${CUDA_cusolver_LIBRARY} ${CUDNN_HOME}/lib/libcudnn.so)
  endif()

  # create cuda_config.h
  configure_file(cuda_config.h.in ${tensorflow_source_dir}/third_party/gpus/cuda/cuda_config.h)
  install(FILES ${tensorflow_source_dir}/third_party/gpus/cuda/cuda_config.h DESTINATION "include/tensorflow/cuda")

  if(UNIX)
   execute_process(COMMAND ln -snf "${CUDA_TOOLKIT_TARGET_DIR}/extras" "${tensorflow_source_dir}/third_party/gpus/cuda/")
  endif(UNIX)

  # tf assumes in various places header files to be in cuda/include. On windows the cuda sdk
  # installs them under cuda/version/include and to avoid that we need to change tf we copy a
  # few files to cuda/include
  FILE(COPY
    ${CUDA_TOOLKIT_TARGET_DIR}/include/cuda.h ${CUDA_TOOLKIT_TARGET_DIR}/include/cuComplex.h
    ${CUDA_TOOLKIT_TARGET_DIR}/include/cublas_v2.h ${CUDNN_HOME}/include/cudnn.h
    ${CUDA_TOOLKIT_TARGET_DIR}/include/cufft.h ${CUDA_TOOLKIT_TARGET_DIR}/include/curand.h
    ${CUDA_TOOLKIT_TARGET_DIR}/include/cuda_runtime_api.h
    ${CUDA_TOOLKIT_TARGET_DIR}/include/cusolverDn.h
    DESTINATION ${tensorflow_source_dir}/third_party/gpus/cuda/include
  )
  include_directories(${tensorflow_source_dir}/third_party/gpus)
  FILE(COPY
    ${NCCL_INCLUDES}
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/external/nccl_archive/src/
  )
endif(tensorflow_ENABLE_GPU)

# Find python executable
include(FindPythonInterp)
if(NOT ${PYTHONINTERP_FOUND})
    message(FATAL_ERROR "CMake was unable to find a python interpreter.")
endif()

# Let's get to work!
include(tf_core_framework.cmake)
# NOTE: Disabled until issue #3996 is fixed.
# include(tf_stream_executor.cmake)
if (tensorflow_ENABLE_GPU)
  include(tf_stream_executor.cmake)
endif()

include(tf_core_cpu.cmake)
include(tf_core_ops.cmake)
include(tf_core_direct_session.cmake)
include(tf_core_kernels.cmake)
if(tensorflow_ENABLE_GRPC_SUPPORT)
  include(tf_core_distributed_runtime.cmake)
endif()
if(tensorflow_ENABLE_MPI)
  include(tf_core_mpi.cmake)
endif()
# We include tf_cc_ops first, because tf_c depends on tf_cc.
include(tf_cc_ops.cmake)
include(tf_c.cmake)
include(tf_core_profiler.cmake)
if(tensorflow_BUILD_CC_EXAMPLE)
  include(tf_tutorials.cmake)
  include(tf_label_image_example.cmake)
endif()
include(tf_tools.cmake)
include(tf_grappler.cmake)
if(tensorflow_BUILD_PYTHON_BINDINGS)
  include(tf_python.cmake)
endif()
include(tf_shared_lib.cmake)
if(tensorflow_BUILD_CC_TESTS OR tensorflow_BUILD_PYTHON_TESTS)
  include(tf_tests.cmake)
endif()

EXPORT(PACKAGE Tensorflow)
SET(TENSORFLOW_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/include/tensorflow")
CONFIGURE_FILE(TensorflowConfig.cmake.in "${PROJECT_BINARY_DIR}/TensorflowConfig.cmake")
INSTALL(FILES "${PROJECT_BINARY_DIR}/TensorflowConfig.cmake" DESTINATION "share/tensorflow/cmake")

MESSAGE(STATUS "<<< Gentoo configuration >>>
Build type      ${CMAKE_BUILD_TYPE}
Install path    ${CMAKE_INSTALL_PREFIX}
Compiler flags:
C               ${CMAKE_C_FLAGS}
C++             ${CMAKE_CXX_FLAGS}
Linker flags:
Executable      ${CMAKE_EXE_LINKER_FLAGS}
Module          ${CMAKE_MODULE_LINKER_FLAGS}
Shared          ${CMAKE_SHARED_LINKER_FLAGS}\n")
