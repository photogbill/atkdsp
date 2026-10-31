@echo off
setlocal enableextensions
rem  Build atkdsp.dll into bin\ and run the C smoke test. Exit 0 = the DLL
rem  exists and the smoke test passed.
rem
rem  Needs the MSVC build tools (the same ones ATK's install.bat uses for
rem  llama-cpp-python). Run from a "x64 Native Tools Command Prompt", or let
rem  this script find vcvars64.bat itself.
rem
rem  CMake is used when found - %ATKDSP_CMAKE% if set (ATK's get_atkdsp.bat
rem  points it at envs\atk_core), then PATH, then ATK's env in the two places
rem  a checkout can sit (vendor\atkdsp inside ATK, or a sibling of ATK).
rem  Without CMake, cl.exe is driven directly.
cd /d "%~dp0"
if not exist bin mkdir bin

rem  cl.exe on PATH is not enough: INCLUDE and LIB must be set too, or
rem  every compile fails with "Cannot open include file: 'math.h'"
rem  (2026-10-29, Bill's PC - cl.exe on PATH, no developer environment).
rem  vcvars64 sets all three, so it is skipped only when all three are there.
where cl >nul 2>nul
if errorlevel 1 goto :find_vs
if defined INCLUDE if defined LIB goto :have_cl
:find_vs
rem  NOT inside a parenthesised block: the ")" in %ProgramFiles(x86)% closes
rem  the block early (ATK's install.bat learned this the hard way).
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto :have_cl
for /f "usebackq tokens=*" %%p in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do set "VSPATH=%%p"
if defined VSPATH call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
:have_cl
where cl >nul 2>nul
if errorlevel 1 (
  echo atkdsp: no C compiler found. Install the MSVC build tools ^(ATK's get_buildtools.bat^) and retry.
  exit /b 1
)

set "CMAKE="
if defined ATKDSP_CMAKE if exist "%ATKDSP_CMAKE%" set "CMAKE=%ATKDSP_CMAKE%"
if not defined CMAKE (
  where cmake >nul 2>nul
  if not errorlevel 1 set "CMAKE=cmake"
)
if not defined CMAKE if exist "..\..\envs\atk_core\Scripts\cmake.exe" set "CMAKE=..\..\envs\atk_core\Scripts\cmake.exe"
if not defined CMAKE if exist "..\ATK\envs\atk_core\Scripts\cmake.exe" set "CMAKE=..\ATK\envs\atk_core\Scripts\cmake.exe"

if defined CMAKE goto :with_cmake

echo atkdsp: cmake not found, compiling with cl.exe directly
cl /nologo /O2 /W3 /fp:precise /arch:AVX2 /openmp /DATKDSP_BUILD=1 /Iinclude /Ivendor\pocketfft ^
   src\*.c vendor\pocketfft\pocketfft.c /LD /Fe:bin\atkdsp.dll /Fo:bin\ || exit /b 1
cl /nologo /O2 /Iinclude tests\test_smoke.c /Fe:bin\atkdsp_smoke.exe /Fo:bin\ bin\atkdsp.lib || exit /b 1
rem the ring's stress test and Bill's bench (tests\test_ring_stress.c)
cl /nologo /O2 /Iinclude tests\test_ring_stress.c /Fe:bin\atkdsp_ring_stress.exe /Fo:bin\ bin\atkdsp.lib || exit /b 1
rem the channel splitter's bench (bench\bench_pfb.c; pfb_bench.bat runs it)
cl /nologo /O2 /Iinclude bench\bench_pfb.c /Fe:bin\atkdsp_bench_pfb.exe /Fo:bin\ bin\atkdsp.lib || exit /b 1
del /q bin\*.obj bin\*.exp 2>nul
goto :smoke

:with_cmake
rem  Ninja (single-config, writes straight into bin\) when it is available -
rem  ATK's env has it - otherwise the Visual Studio generator, which puts the
rem  outputs under a Release\ subfolder that is copied up afterwards.
rem
rem  atkdsp is built with MSVC. Two things stop CMake picking another compiler:
rem   - a FRESH build\ is told to use cl. Without it, Ninja took the first C
rem     compiler on PATH: Strawberry Perl's MinGW gcc (2026-10-31, the vendored
rem     copy in ATK's test install), which could not link the DLL;
rem   - a build\ whose cache names another compiler is started again. CMake
rem     keeps the compiler it chose first for as long as build\ exists, so
rem     that copy failed the same way on every update.
if not exist build\CMakeCache.txt goto :cmake_configure
findstr /r /i /c:"^CMAKE_C_COMPILER:[A-Z]*=" build\CMakeCache.txt >nul 2>nul
if errorlevel 1 goto :cmake_configure
findstr /r /i /c:"^CMAKE_C_COMPILER:[A-Z]*=.*cl\.exe" build\CMakeCache.txt >nul 2>nul
if not errorlevel 1 goto :cmake_configure
echo atkdsp: build\ was set up with another compiler, not MSVC - starting it again
rd /s /q build
:cmake_configure
set "CC_ARG="
if not exist build\CMakeCache.txt set "CC_ARG=-DCMAKE_C_COMPILER=cl"
"%CMAKE%" -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release %CC_ARG% >nul 2>nul
if errorlevel 1 (
  if exist build rd /s /q build
  "%CMAKE%" -S . -B build -DCMAKE_BUILD_TYPE=Release || exit /b 1
)
"%CMAKE%" --build build --config Release || exit /b 1
if exist bin\Release\atkdsp.dll copy /y bin\Release\atkdsp.dll bin\ >nul
if exist bin\Release\atkdsp_smoke.exe copy /y bin\Release\atkdsp_smoke.exe bin\ >nul
if exist build\Release\atkdsp.dll copy /y build\Release\atkdsp.dll bin\ >nul
if exist build\Release\atkdsp_smoke.exe copy /y build\Release\atkdsp_smoke.exe bin\ >nul
if exist bin\Release\atkdsp_ring_stress.exe copy /y bin\Release\atkdsp_ring_stress.exe bin\ >nul
if exist build\Release\atkdsp_ring_stress.exe copy /y build\Release\atkdsp_ring_stress.exe bin\ >nul
if exist bin\Release\atkdsp_bench_pfb.exe copy /y bin\Release\atkdsp_bench_pfb.exe bin\ >nul
if exist build\Release\atkdsp_bench_pfb.exe copy /y build\Release\atkdsp_bench_pfb.exe bin\ >nul

:smoke
if not exist bin\atkdsp.dll (
  echo atkdsp: build finished but bin\atkdsp.dll is not there.
  exit /b 1
)
bin\atkdsp_smoke.exe
exit /b %ERRORLEVEL%
