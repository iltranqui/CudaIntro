@echo off
:: Set the path to Visual Studio and MSVC
set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community"
set "MSVC_PATH=%VS_PATH%\VC\Tools\MSVC\14.37.32822"

:: Add Visual Studio and MSVC paths to PATH
set "PATH=%MSVC_PATH%\bin\Hostx64\x64;%PATH%"
set "PATH=%VS_PATH%\Common7\IDE;%PATH%"
set "PATH=%VS_PATH%\VC\Auxiliary\Build;%PATH%"

:: Set up Visual C++ environment variables
set "INCLUDE=%MSVC_PATH%\include;%INCLUDE%"
set "LIB=%MSVC_PATH%\lib\x64;%LIB%"

:: Set OpenCV paths
set "OPENCV_DIR=c:\src\vcpkg\installed\x64-windows"
set "INCLUDE=%OPENCV_DIR%\include;%INCLUDE%"

echo Compiling MNIST CNN...

:: Call Visual Studio environment setup
call "%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat"

:: Compile with nvcc
nvcc -o mnist_cnn.exe mnist_cnn.cu mnist_cnn_main.cu mnist_kernels.cu ^
    -I"%OPENCV_DIR%\include" ^
    -I"%OPENCV_DIR%\include\opencv4" ^
    -L"%OPENCV_DIR%\lib" ^
    -lopencv_world4 ^
    -lcudart -lcuda -lcudadevrt ^
    -gencode arch=compute_89,code=sm_89 ^
    -O3 -std=c++20 ^
    -diag-suppress 177

if %ERRORLEVEL% NEQ 0 (
    echo Compilation failed!
) else (
    echo Compilation successful!
)
