@echo off
rem  pfb_bench.bat - the channel splitter (atkdsp_pfb) on this PC: how many
rem  times real time it runs, one radio from 2.4 to 56 MS/s (ATK's trunking
rem  grid, ~100 kHz channels), the survey's 64-channel scale, and the Kraken
rem  DF bank (192 channels x 5 elements), at 1, 2, 4, ... threads up to every
rem  core. 1.00 means the worker is exactly saturated. Double-click it after
rem  build.bat (or ATK's build.bat /only atkdsp). Takes a minute or two. The
rem  result is printed and saved in bench\pfb_bench_result.txt beside this
rem  file (nothing goes to C:). Optional: pfb_bench.bat MAX_THREADS
setlocal
cd /d "%~dp0"
if not exist bin\atkdsp_bench_pfb.exe (
  echo The bench program is not built yet - run build.bat first.
  pause
  exit /b 1
)
if not exist bench mkdir bench
echo Running the channel-splitter bench. Please wait...
bin\atkdsp_bench_pfb.exe %1 > bench\pfb_bench_result.txt 2>&1
set "RC=%ERRORLEVEL%"
type bench\pfb_bench_result.txt
if "%RC%"=="0" (echo Done - send bench\pfb_bench_result.txt to Claude if you want it read.) else (echo The bench FAILED - send bench\pfb_bench_result.txt to Claude.)
pause
exit /b %RC%
