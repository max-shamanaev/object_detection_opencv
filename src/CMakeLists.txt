cmake_minimum_required(VERSION 3.5)

project(object_detection VERSION 1)

set(CMAKE_CXX_STANDARD 17)

# CUDA
find_package(CUDAToolkit)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# !CUDA

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})
# !OpenCV

set(PROJECT_SOURCES
    main.cpp

    objectDetection.h
    objectDetection.cpp
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY $<1:../bin>)

add_executable(object_detection ${PROJECT_SOURCES})
target_link_libraries(
    object_detection  
    PRIVATE ${OpenCV_LIBS}
)