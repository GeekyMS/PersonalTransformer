cmake_minimum_required(VERSION 3.20)
project(personal_transformer_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

add_compile_options(-Wall -Wextra)

# Everything in src/ becomes the "model" library: Tensor, allocator, ops,
# autograd. Test binaries and the trainer link against this.
file(GLOB MODEL_SOURCES CONFIGURE_DEPENDS src/*.cpp)

if(MODEL_SOURCES)
  add_library(model STATIC ${MODEL_SOURCES})
  target_include_directories(model PUBLIC src include)

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
