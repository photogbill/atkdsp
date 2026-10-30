cmake_minimum_required(VERSION 3.16)
project(atkdsp C)

# atkdsp — the Analyst Toolkit's native DSP kernels. One shared library, one
# header (include/atkdsp.h) that IS the ABI, no dependencies beyond libm and
# an optional OpenMP. See README.md.

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(ATKDSP_OPENMP "Parallel batched FFT via OpenMP" ON)
option(ATKDSP_AVX2   "Compile the kernels for AVX2 (x86-64 v3)" ON)
option(ATKDSP_TESTS  "Build the C smoke test" ON)

set(ATKDSP_SOURCES
  src/atkdsp.c src/convert.c src/nco.c src/fir.c src/resample.c
  src/fft.c src/demod.c src/display.c src/detect.c src/design.c src/ddc.c
  src/lte.c src/lte_pbch.c src/lte_prach.c src/lte_turbo.c src/lte_sib1.c
  src/lte_sib1_phy.c src/lte_si.c src/fsk.c src/dmr_sync.c
  src/stat.c src/bits.c src/scramble.c src/fec_conv.c src/fec_rs.c src/fec_blk.c
  src/crc.c src/crc_recover.c src/gf2.c src/coderec.c
  src/cfar.c src/burst.c src/ring.c
  src/pulses.c src/cyclic.c src/tone_id.c
  src/dsss_det.c src/dsss_codes.c src/dsss_code.c src/dsss_acq.c src/css.c src/xspec.c
  src/channelizer.c src/undertow.c src/psync.c src/lagscan.c src/video.c
  vendor/pocketfft/pocketfft.c)

add_library(atkdsp SHARED ${ATKDSP_SOURCES})
target_include_directories(atkdsp
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vendor/pocketfft)
target_compile_definitions(atkdsp PRIVATE ATKDSP_BUILD=1)
set_target_properties(atkdsp PROPERTIES
  C_VISIBILITY_PRESET hidden
  OUTPUT_NAME atkdsp
  PREFIX ""          # atkdsp.dll / atkdsp.so — one name everywhere
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

if(MSVC)
  # NOT /fp:fast: NaN means "unmeasured" in the stitch and display kernels,
  # and fast-math lets the compiler delete the tests for it.
  target_compile_options(atkdsp PRIVATE /O2 /W3 /fp:precise)
  if(ATKDSP_AVX2)
    target_compile_options(atkdsp PRIVATE /arch:AVX2)
  endif()
else()
  # NOT -ffast-math (see above); -fno-math-errno alone keeps sqrtf inline.
  target_compile_options(atkdsp PRIVATE -O3 -Wall -Wextra -fno-math-errno)
  if(ATKDSP_AVX2)
    target_compile_options(atkdsp PRIVATE -mavx2 -mfma)
  endif()
  target_link_libraries(atkdsp PRIVATE m)
endif()

if(ATKDSP_OPENMP)
  find_package(OpenMP)
  if(OpenMP_C_FOUND)
    target_link_libraries(atkdsp PRIVATE OpenMP::OpenMP_C)
  else()
    message(STATUS "atkdsp: OpenMP not found — batched FFT runs single-threaded")
  endif()
endif()

if(ATKDSP_TESTS)
  add_executable(atkdsp_smoke tests/test_smoke.c)
  target_link_libraries(atkdsp_smoke PRIVATE atkdsp)
  if(NOT MSVC)
    target_link_libraries(atkdsp_smoke PRIVATE m)
  endif()
  set_target_properties(atkdsp_smoke PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
  # the ring's stress test and Bill's bench (tests/test_ring_stress.c)
  find_package(Threads)
  add_executable(atkdsp_ring_stress tests/test_ring_stress.c)
  target_link_libraries(atkdsp_ring_stress PRIVATE atkdsp ${CMAKE_THREAD_LIBS_INIT})
  if(NOT MSVC)
    target_compile_definitions(atkdsp_ring_stress PRIVATE _POSIX_C_SOURCE=200112L)
  endif()
  set_target_properties(atkdsp_ring_stress PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
  # the channel splitter's bench (bench/bench_pfb.c; pfb_bench.bat runs it)
  add_executable(atkdsp_bench_pfb bench/bench_pfb.c)
  target_link_libraries(atkdsp_bench_pfb PRIVATE atkdsp)
  if(NOT MSVC)
    target_link_libraries(atkdsp_bench_pfb PRIVATE m)
  endif()
  set_target_properties(atkdsp_bench_pfb PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
  enable_testing()
  add_test(NAME smoke COMMAND atkdsp_smoke)
  add_test(NAME ring_stress COMMAND atkdsp_ring_stress 1)
endif()

install(TARGETS atkdsp DESTINATION bin)
install(FILES include/atkdsp.h DESTINATION include)
