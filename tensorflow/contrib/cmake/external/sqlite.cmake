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

set(sqlite_INCLUDE_DIR "")
set(sqlite_URL http://www.sqlite.org/2017/sqlite-amalgamation-3200000.zip)
set(sqlite_HASH SHA256=208780b3616f9de0aeb50822b7a8f5482f6515193859e91ed61637be6ad74fd4)
set(sqlite_BUILD ${CMAKE_CURRENT_BINARY_DIR}/sqlite/src/sqlite)
set(sqlite_INSTALL ${CMAKE_CURRENT_BINARY_DIR}/sqlite/install)

if(WIN32)
  set(sqlite_STATIC_LIBRARIES ${sqlite_INSTALL}/lib/sqlite.lib)
else()
  #set(sqlite_STATIC_LIBRARIES ${sqlite_INSTALL}/lib/libsqlite.a)
  set(sqlite_STATIC_LIBRARIES /mnt/bwpy/single/usr/lib/libsqlite3.so)
endif()

