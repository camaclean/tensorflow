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

set(zlib_INCLUDE_DIR "")
set(ZLIB_URL https://github.com/madler/zlib)
set(ZLIB_BUILD ${CMAKE_CURRENT_BINARY_DIR}/zlib/src/zlib)
set(ZLIB_INSTALL ${CMAKE_CURRENT_BINARY_DIR}/zlib/install)
set(ZLIB_TAG 50893291621658f355bc5b4d450a8d06a563053d)

if(WIN32)
  set(zlib_STATIC_LIBRARIES
      debug ${CMAKE_CURRENT_BINARY_DIR}/zlib/install/lib/zlibstaticd.lib
      optimized ${CMAKE_CURRENT_BINARY_DIR}/zlib/install/lib/zlibstatic.lib)
else()
  set(zlib_STATIC_LIBRARIES
      /mnt/bwpy/single/usr/lib/libz.so)
endif()

