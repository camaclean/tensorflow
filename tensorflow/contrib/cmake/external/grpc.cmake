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
include (ExternalProject)

set(GRPC_INCLUDE_DIRS /mnt/bwpy/single/usr/include/grpc)
set(GRPC_URL https://github.com/grpc/grpc.git)
set(GRPC_BUILD ${CMAKE_CURRENT_BINARY_DIR}/grpc/src/grpc)
set(GRPC_TAG 781fd6f6ea03645a520cd5c675da67ab61f87e4b)

find_package(gRPC)
set(GRPC_BUILD /mnt/bwpy/single/usr/bin)

set(grpc_STATIC_LIBRARIES
    gRPC::grpc++_unsecure
    gRPC::grpc_unsecure
    gRPC::gpr
    /mnt/bwpy/single/usr/lib/libcares.so
)
