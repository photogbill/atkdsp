#!/bin/sh
# Build atkdsp.so into bin/ and run the C smoke test. Needs a C compiler;
# CMake is used when present, otherwise a direct gcc/clang line.
set -e
cd "$(dirname "$0")"
mkdir -p bin
if command -v cmake >/dev/null 2>&1; then
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build build --config Release
else
  CC=${CC:-cc}
  $CC -std=c99 -O3 -Wall -Wextra -fno-math-errno -fopenmp -fPIC -shared -fvisibility=hidden \
      -DATKDSP_BUILD=1 -Iinclude -Ivendor/pocketfft src/*.c vendor/pocketfft/pocketfft.c \
      -o bin/atkdsp.so -lm
  $CC -std=c99 -O2 -Iinclude tests/test_smoke.c -o bin/atkdsp_smoke -Lbin -l:atkdsp.so -lm -Wl,-rpath,'$ORIGIN'
fi
./bin/atkdsp_smoke
