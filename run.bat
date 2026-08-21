@echo off
REM ============================================================
REM  dlfc-code 运行器：双击运行某章脚本（默认第 1 章）
REM  用法: run.bat            -> 运行第 1 章
REM        run.bat ch05       -> 运行第 5 章（按文件名前缀匹配）
REM ============================================================
setlocal
set ROOT=%~dp0

REM 优先用 PATH 里的 python，找不到再退回 C:\Python314
set PY=python
%PY% --version >nul 2>&1
if errorlevel 1 set PY=C:\Python314\python.exe

if "%~1"=="" (
    set TARGET=ch01
) else (
    set TARGET=%~1
)

for /f "delims=" %%f in ('dir /b /s "%ROOT%chapters\%TARGET%*.py" 2^>nul') do (
    echo [dlfc-code] 运行 %%f
    "%PY%" "%%f"
    goto :done
)

echo [dlfc-code] 未找到章节 %TARGET% 的脚本，请检查参数（如 run.bat ch02）
:done
pause
