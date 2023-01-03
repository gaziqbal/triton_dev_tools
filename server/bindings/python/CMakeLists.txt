cmake_minimum_required(VERSION 3.18)

# PyBind11 dependency
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.10.2
    GIT_SHALLOW ON    
    )
FetchContent_MakeAvailable(pybind11)    

# PyBind11 configuration
set(Python3_FIND_VIRTUALENV ONLY)
set(Python3_FIND_STRATEGY LOCATION)
set(Python3_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/.venv)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)


message("CMAKE_CURRENT_SOURCE_DIR:${CMAKE_CURRENT_SOURCE_DIR}")
message("Python3_VERSION:${Python3_VERSION}")
message("Python3_EXECUTABLE:${Python3_EXECUTABLE}")
message("Python3_LIBRARIES:${Python3_LIBRARIES}")

# find_package(pybind11 CONFIG REQUIRED)
# add_subdirectory(extern/pybind11)

# Python Bindings Project
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