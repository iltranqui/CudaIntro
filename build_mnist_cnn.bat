@echo off
echo Building mnist_cnn project with CMake...

REM Create build directory if it doesn't exist
if not exist build mkdir build

REM Navigate to build directory
cd build

REM Configure CMake
cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_PREFIX_PATH=c:/src/vcpkg/installed/x64-windows ..

REM Build only mnist_cnn
cmake --build . --config Release --target mnist_cnn

REM Copy executable to root directory
echo Copying executable to root directory...
copy Release\mnist_cnn.exe ..\

echo Build complete!
cd ..
