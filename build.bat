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

where cl >nul 2>nul
if not errorlevel 1 goto :have_cl
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
del /q bin\*.obj bin\*.exp 2>nul
goto :smoke

:with_cmake
rem  Ninja (single-config, writes straight into bin\) when it is available -
rem  ATK's env has it - otherwise the Visual Studio generator, which puts the
rem  outputs under a Release\ subfolder that is copied up afterwards.
"%CMAKE%" -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release >nul 2>nul
if errorlevel 1 (
  if exist build rd /s /q build
  "%CMAKE%" -S . -B build -DCMAKE_BUILD_TYPE=Release || exit /b 1
)
"%CMAKE%" --build build --config Release || exit /b 1
if exist bin\Release\atkdsp.dll copy /y bin\Release\atkdsp.dll bin\ >nul
if exist bin\Release\atkdsp_smoke.exe copy /y bin\Release\atkdsp_smoke.exe bin\ >nul
if exist build\Release\atkdsp.dll copy /y build\Release\atkdsp.dll bin\ >nul
if exist build\Release\atkdsp_smoke.exe copy /y build\Release\atkdsp_smoke.exe bin\ >nul

:smoke
if not exist bin\atkdsp.dll (
  echo atkdsp: build finished but bin\atkdsp.dll is not there.
  exit /b 1
)
bin\atkdsp_smoke.exe
exit /b %ERRORLEVEL%
