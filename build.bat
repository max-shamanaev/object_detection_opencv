rmdir /s /q build
mkdir build
cd build
cmake -G "Visual Studio 16 2019" "../src"
cmake --build . --config Release
