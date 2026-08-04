#!/usr/bin/env bash
# 编译 rosetta_env —— 需要先有一份编好的 RosettaStone。
#
#   ./rosetta/build.sh                 # 用 ~/Documents/RosettaStone
#   ROSETTA_ROOT=/path ./rosetta/build.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
ROSETTA_ROOT="${ROSETTA_ROOT:-$HOME/Documents/RosettaStone}"
PYTHON="${PYTHON:-$REPO/.venv/bin/python3.12}"

if [ ! -d "$ROSETTA_ROOT" ]; then
    echo "找不到 RosettaStone：$ROSETTA_ROOT" >&2
    echo "先 git clone --depth 1 https://github.com/utilForever/RosettaStone.git \"$ROSETTA_ROOT\"" >&2
    exit 1
fi

# 上游还没编过就先编它。需要 vcpkg（brew install vcpkg 或 ~/vcpkg）。
if [ ! -f "$ROSETTA_ROOT/build/lib/libRosettaStone.a" ]; then
    : "${VCPKG_ROOT:=$HOME/vcpkg}"
    if [ ! -f "$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" ]; then
        echo "找不到 vcpkg：$VCPKG_ROOT（用 VCPKG_ROOT 指过去）" >&2
        exit 1
    fi
    echo ">>> 编译 RosettaStone（第一次大概 5~10 分钟）"
    cmake -S "$ROSETTA_ROOT" -B "$ROSETTA_ROOT/build" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
    cmake --build "$ROSETTA_ROOT/build" --config Release
fi

echo ">>> 编译 rosetta_env 绑定"
cmake -S "$HERE/native" -B "$HERE/native/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DROSETTA_ROOT="$ROSETTA_ROOT" \
    -DPython_EXECUTABLE="$PYTHON"
cmake --build "$HERE/native/build"

echo ">>> 完成"
ls -la "$HERE"/rosetta_env*.so
