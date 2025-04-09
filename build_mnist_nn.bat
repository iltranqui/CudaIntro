@echo off
echo Building mnist_nn project with CMake...

REM Create build directory if it doesn't exist
if not exist build mkdir build

REM Navigate to build directory
cd build

REM Configure CMake
cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_PREFIX_PATH=c:/src/vcpkg/installed/x64-windows ..

REM Build only mnist_nn
cmake --build . --config Release --target mnist_nn

REM Copy executable to root directory
echo Copying executable to root directory...
copy Release\mnist_nn.exe ..\

echo Build complete!
cd ..
