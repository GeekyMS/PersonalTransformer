cmake_minimum_required(VERSION 3.20)
project(personal_transformer_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

add_compile_options(-Wall -Wextra)

# --constraint=amd7763 in train.slurm targets AMD EPYC 7763 (Zen3). GCC 9.4
# (the module train.slurm loads) doesn't know -march=znver3 yet (added in
# GCC 10), but znver2 is recognized and still enables the AVX2/FMA ISA
# extensions Zen3 supports -- just without Zen3-specific instruction tuning.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options(-march=znver2 -mtune=znver2)
endif()

find_package(OpenMP)

option(USE_BLAS "Use BLAS for CPU matmul forward and backward" ON)
if(USE_BLAS)
  if(APPLE)
    set(BLAS_DEFAULT_VENDOR Apple)
  else()
    set(BLAS_DEFAULT_VENDOR OpenBLAS)
  endif()
  set(BLA_VENDOR ${BLAS_DEFAULT_VENDOR} CACHE STRING "BLAS library vendor")
  find_package(BLAS REQUIRED)

  if(BLA_VENDOR STREQUAL "Apple")
    set(CBLAS_HEADER "Accelerate/Accelerate.h")
  else()
    set(CBLAS_HEADER "cblas.h")
    foreach(blas_library IN LISTS BLAS_LIBRARIES)
      get_filename_component(blas_dir "${blas_library}" DIRECTORY)
      list(APPEND CBLAS_HINTS "${blas_dir}/../include")
    endforeach()
    find_path(CBLAS_INCLUDE_DIR cblas.h HINTS ${CBLAS_HINTS} ENV CPATH PATH_SUFFIXES openblas NO_DEFAULT_PATH)
    find_path(CBLAS_INCLUDE_DIR cblas.h PATH_SUFFIXES openblas REQUIRED)
  endif()

  # FindBLAS checks the Fortran interface; also check the C interface we use.
  include(CheckCXXSourceCompiles)
  include(CMakePushCheckState)
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_LIBRARIES BLAS::BLAS)
  set(CMAKE_REQUIRED_INCLUDES ${CBLAS_INCLUDE_DIR})
  if(BLA_VENDOR STREQUAL "Apple")
    set(CMAKE_REQUIRED_DEFINITIONS -DACCELERATE_NEW_LAPACK)
  endif()
  check_cxx_source_compiles("#include <${CBLAS_HEADER}>
    int main() {
      float a = 1.0f, b = 1.0f, c = 0.0f;
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, 1, 1,
                  1.0f, &a, 1, &b, 1, 0.0f, &c, 1);
      return 0;
    }" HAVE_CBLAS)
  cmake_pop_check_state()
  if(NOT HAVE_CBLAS)
    message(FATAL_ERROR "Selected BLAS does not provide cblas_sgemm. Load OpenBLAS or set BLA_VENDOR and CBLAS_INCLUDE_DIR.")
  endif()
  message(STATUS "CPU matmul: BLAS (${BLA_VENDOR})")
else()
  message(STATUS "CPU matmul: handwritten OpenMP")
endif()

# Everything in src/ becomes the "model" library: Tensor, allocator, ops,
# autograd. Test binaries and the trainer link against this.
file(GLOB MODEL_SOURCES CONFIGURE_DEPENDS src/*.cpp)

if(MODEL_SOURCES)
  add_library(model STATIC ${MODEL_SOURCES})
  target_include_directories(model PUBLIC src include)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(model PUBLIC OpenMP::OpenMP_CXX)
  endif()
  if(USE_BLAS)
    target_compile_definitions(model PRIVATE USE_BLAS)
    target_link_libraries(model PUBLIC BLAS::BLAS)
    if(BLA_VENDOR STREQUAL "Apple")
      target_compile_definitions(model PRIVATE USE_ACCELERATE ACCELERATE_NEW_LAPACK)
    else()
      target_include_directories(model PRIVATE ${CBLAS_INCLUDE_DIR})
    endif()
  endif()

  # Each test/op_*.cpp becomes its own standalone binary: loads input dumps,
  # runs one op, writes the output dump. See test/README.md for the contract.
  file(GLOB OP_TESTS CONFIGURE_DEPENDS test/op_*.cpp)
  foreach(test_src ${OP_TESTS})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE model)
  endforeach()
else()
  message(STATUS "src/ is empty — add tensor.cpp / allocator.cpp / ops.cpp etc. to build the model lib and op tests.")
endif()

# The standalone trainer: no Python at runtime.
if(MODEL_SOURCES AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/train.cpp")
  add_executable(train src/train.cpp)
  target_link_libraries(train PRIVATE model)
endif()
