@echo off
echo Setting up Visual Studio environment...

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

:: Set OpenCV paths
set "OPENCV_DIR=c:\src\vcpkg\installed\x64-windows"

echo Compiling MNIST Neural Network...
nvcc -o mnist_nn.exe mnist_nn.cu ^
    -I"%OPENCV_DIR%\include" ^
    -I"%OPENCV_DIR%\include\opencv4" ^
    -L"%OPENCV_DIR%\lib" ^
    -lopencv_world4 ^
    -O3 -std=c++14 ^
    --verbose

if %ERRORLEVEL% EQU 0 (
    echo Compilation successful!
    echo Running the program...
    mnist_nn.exe
) else (
    echo Compilation failed with error code %ERRORLEVEL%.
)

pause
