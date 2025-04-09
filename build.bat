@echo off
echo Building MNIST projects with CMake...

REM Create build directory if it doesn't exist
if not exist build mkdir build

REM Navigate to build directory
cd build

REM Configure CMake
cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_PREFIX_PATH=c:/src/vcpkg/installed/x64-windows ..

REM Build both projects
cmake --build . --config Release

REM Copy executables to root directory
echo Copying executables to root directory...
copy Release\mnist_nn.exe ..\
copy Release\mnist_cnn.exe ..\

echo Build complete!
cd ..
