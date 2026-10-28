@echo off
rem  ring_bench.bat - the ring bench on this PC (docs\ATKDSP_PLAN.md section A3):
rem  40 MSPS of ci16 (160 MB/s) for ten minutes, one producer and three
rem  consumers (fast, slow, stalled), 1 MiB blocks, a 512 MiB ring.
rem  Double-click it after build.bat. The result is printed and saved in
rem  bench\ring_bench_result.txt beside this file (nothing goes to C:).
rem  Optional: ring_bench.bat SECONDS MB_PER_S BLOCK_KIB NBLOCKS
setlocal
cd /d "%~dp0"
if not exist bin\atkdsp_ring_stress.exe (
  echo The bench program is not built yet - run build.bat first.
  pause
  exit /b 1
)
set "SECS=%~1"
if "%SECS%"=="" set "SECS=600"
set "RATE=%~2"
if "%RATE%"=="" set "RATE=160"
set "BLK=%~3"
if "%BLK%"=="" set "BLK=1024"
set "NB=%~4"
if "%NB%"=="" set "NB=512"
if not exist bench mkdir bench
echo Running the ring bench: %SECS% s at %RATE% MB/s, %BLK% KiB x %NB% blocks. Please wait...
bin\atkdsp_ring_stress.exe %SECS% %RATE% %BLK% %NB% > bench\ring_bench_result.txt 2>&1
set "RC=%ERRORLEVEL%"
type bench\ring_bench_result.txt
if "%RC%"=="0" (echo The ring held: every check passed.) else (echo The ring bench FAILED - send bench\ring_bench_result.txt to Claude.)
pause
exit /b %RC%
