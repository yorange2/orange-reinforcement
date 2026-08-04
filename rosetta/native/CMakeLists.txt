cmake_minimum_required(VERSION 3.24)
project(rosetta_env LANGUAGES CXX)

# 链接一份已经编好的 RosettaStone。默认位置是仓库的同级目录，
# 需要时用 -DROSETTA_ROOT=... 覆盖。
if (NOT ROSETTA_ROOT)
    set(ROSETTA_ROOT "$ENV{HOME}/Documents/RosettaStone")
endif()
set(ROSETTA_BUILD "${ROSETTA_ROOT}/build")
set(VCPKG_TREE "${ROSETTA_BUILD}/vcpkg_installed/arm64-osx")

foreach(path "${ROSETTA_ROOT}/Includes" "${ROSETTA_BUILD}/lib/libRosettaStone.a")
    if (NOT EXISTS "${path}")
        message(FATAL_ERROR "找不到 ${path}——先把 RosettaStone 编出来（见 rosetta/README.md）")
    endif()
endforeach()

set(CMAKE_CXX_STANDARD 23)          # 必须和 RosettaStone 一致，否则 ABI 对不上
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 不加这句 pybind11 会走过时的 FindPythonInterp，抓到系统的 python3.9 而不是 venv 的 3.12
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED PATHS "${VCPKG_TREE}/share/pybind11" NO_DEFAULT_PATH)

pybind11_add_module(rosetta_env bindings.cpp actions.cpp)

# 直接产到 rosetta/ 包目录下，这样 `from rosetta import rosetta_env` 就能用
set_target_properties(rosetta_env PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/..")

target_include_directories(rosetta_env PRIVATE
    "${ROSETTA_ROOT}/Includes"
    "${VCPKG_TREE}/include")

target_link_libraries(rosetta_env PRIVATE "${ROSETTA_BUILD}/lib/libRosettaStone.a")
