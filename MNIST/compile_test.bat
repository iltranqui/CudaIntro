@echo off
:: Set the path to Visual Studio and MSVC
set "VS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools"
set "MSVC_PATH=%VS_PATH%\VC\Tools\MSVC\14.29.30133"

:: Add Visual Studio and MSVC paths to PATH
set "PATH=%MSVC_PATH%\bin\Hostx64\x64;%PATH%"
set "PATH=%VS_PATH%\Common7\IDE;%PATH%"
set "PATH=%VS_PATH%\VC\Auxiliary\Build;%PATH%"

:: Set up Visual C++ environment variables
set "INCLUDE=%MSVC_PATH%\include;%INCLUDE%"
set "LIB=%MSVC_PATH%\lib\x64;%LIB%"

echo Compiling test program...
cl.exe /EHsc /std:c++17 test_logging.cpp /Fe:test_logging.exe

if %ERRORLEVEL% NEQ 0 (
    echo Compilation failed!
) else (
    echo Compilation successful!
    echo Running test program...
    test_logging.exe
)
