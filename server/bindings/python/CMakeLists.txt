cmake_minimum_required(VERSION 3.18)

# PyBind11
add_subdirectory(extern/pybind11)

#
# gaz_test
#
pybind11_add_module(
  triton
  src/tritonpybind.cc
)

set_target_properties(
  triton
  PROPERTIES
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH ""
)

target_include_directories(
  triton
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(
  triton
  PRIVATE
    triton-developer_tools-server
    triton-core-serverstub
)

install(
  TARGETS triton
  LIBRARY DESTINATION lib
)