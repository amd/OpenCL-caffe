# ########################################################################
# Copyright 2013 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ########################################################################


# Locate an clBLAS library.
#
# Defines the following variables:
#
#   CLBLAS_FOUND - Found the CLBLAS library
#   CLBLAS_INCLUDE_DIRS - Include directories
#
# Also defines the library variables below as normal
# variables.  These contain debug/optimized keywords when
# a debugging library is found.
#
#   CLBLAS_LIBRARIES - libclBLAS
#
# Accepts the following variables as input:
#
#   CLBLAS_ROOT - (as a CMake or environment variable)
#                The root directory of the clBLAS library found
#
#   FIND_LIBRARY_USE_LIB64_PATHS - Global property that controls whether findclBLAS should search for
#                              64bit or 32bit libs
#-----------------------
# Example Usage:
#
#    find_package(clBLAS REQUIRED)
#    include_directories(${CLBLAS_INCLUDE_DIRS})
#
#    add_executable(foo foo.cc)
#    target_link_libraries(foo ${CLBLAS_LIBRARIES})
#
#-----------------------

set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS ON)

find_path(CLBLAS_INCLUDE_DIRS  NAMES clBLAS.h  
    HINTS
        $ENV{CLBLAS_ROOT}/include
    PATHS
        /usr/include
        /usr/local/include
    DOC "clBLAS header file path"
)
mark_as_advanced( CLBLAS_INCLUDE_DIRS )

# Search for 64bit libs if FIND_LIBRARY_USE_LIB64_PATHS is set to true in the global environment, 32bit libs else
get_property( LIB64 GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS )

if( LIB64 )
    find_library( CLBLAS_LIBRARIES
        NAMES clBLAS
        HINTS
            $ENV{CLBLAS_ROOT}/lib64
        DOC "clBLAS dynamic library path"
        PATHS
            /usr/lib
            /usr/local/lib
    )
else( )
    find_library( CLBLAS_LIBRARIES
        NAMES clBLAS
        HINTS
            $ENV{CLBLAS_ROOT}/lib
        DOC "clBLAS dynamic library path"
        PATHS
            /usr/lib
            /usr/local/lib
    )
endif( )
mark_as_advanced( CLBLAS_LIBRARIES )

if (NOT CLBLAS_INCLUDE_DIRS)
   set(CLBLAS_FOUND ON)
endif()

include( FindPackageHandleStandardArgs )
FIND_PACKAGE_HANDLE_STANDARD_ARGS( CLBLAS DEFAULT_MSG CLBLAS_LIBRARIES CLBLAS_INCLUDE_DIRS )

if( NOT CLBLAS_FOUND )
    message( STATUS "FindclBLAS looked for libraries named: clBLAS" )
else ()
    message( STATUS "Found clBLAS  (include: ${CLBLAS_INCLUDE_DIRS}, library: ${CLBLAS_LIBRARIES})")
endif()
