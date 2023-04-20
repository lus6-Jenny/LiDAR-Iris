cmake_minimum_required(VERSION 3.8)

project(LidarIris)

SET(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_FIND_LIBRARY_PREFIXES ${CMAKE_FIND_LIBRARY_PREFIXES} "lib")
set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES} ".so")

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/lib/x86_64-linux-gnu/cmake")

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PythonInterp 3)
find_package(PythonLibs 3)
find_package(pybind11 REQUIRED)

# set(OpenCV_INCLUDE_DIRS "/usr/local/include/opencv4")
set(pybind11_INCLUDE_DIR "/home/lusha/miniconda3/envs/pytorch/lib/python3.7/site-packages/pybind11/include")

include_directories(${PYTHON_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${pybind11_INCLUDE_DIR})
link_directories(${PCL_LIBRARY_DIRS})

set(SOURCE_FILES
    LidarIris/LidarIris.cpp
    LidarIris/LidarIris.h
    LidarIris/fftm/fftm.cpp
    LidarIris/fftm/fftm.hpp)

add_library(LidarIris SHARED ${SOURCE_FILES})
target_include_directories(LidarIris PRIVATE ${Python_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(LidarIris PRIVATE ${Python_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})

pybind11_add_module(lidar_iris ${SOURCE_FILES})
target_include_directories(lidar_iris PUBLIC ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${pybind11_INCLUDE_DIR})
target_include_directories(lidar_iris PRIVATE ${Python_INCLUDE_DIRS})
target_link_libraries(lidar_iris PRIVATE ${Python_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})

# include_directories("./LidarIris/")
# add_executable(demo "demo.cpp" ${SOURCE_FILES})
# target_include_directories(demo PRIVATE ${Python_LIBRARIES} ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
# target_link_libraries(demo ${Python_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})
