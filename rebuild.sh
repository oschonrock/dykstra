#!/usr/bin/env bash

pwd="$(pwd)"
base="$(dirname $(realpath $0))"

profile=${1:-default}

conan_profile="$HOME/.conan/profiles/$profile"
if [[ ! -e "$conan_profile" ]]
then
    echo "Usage: $0 [conan_profile]"
    echo
    echo "conan profile '$conan_profile' does not exist"
    echo
    echo "possible profiles:"
    for p in $(ls "$HOME/.conan/profiles/")
    do
        basename "$p"
    done
    exit 1
fi

echo "using profile $conan_profile"

eval "$(cat "$conan_profile" | egrep '^CC|CXX=.*$')"


cd "$base"
rm -rf build &&                                                \
    mkdir build &&                                             \
    cd build &&                                                \
    conan install .. --profile="$conan_profile" &&             \
    cd .. &&                                                   \
    cmake -DCMAKE_CXX_COMPILER="$CXX"                          \
          -DCMAKE_C_COMPILER="$CC"                             \
          -DCMAKE_EXPORT_COMPILE_COMMANDS=1                    \
          -B build -S .  &&                                    \
    cmake --build build

cd "$pwd"
