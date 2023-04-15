cmake_minimum_required (VERSION 3.8)

SET(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/lib/x86_64-linux-gnu/cmake")
set(OpenCV_DIR "/usr/local/share/OpenCV")

project (LidarIris)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PythonLibs 3.7 REQUIRED)

set(PYTHON_INCLUDE_DIRS "/home/lusha/miniconda3/envs/pytorch/include/python3.7m")

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(pybind11/include)

add_subdirectory(pybind11)
pybind11_add_module(lidar_iris
LidarIris/LidarIris.cpp
)
target_include_directories(lidar_iris PUBLIC
  include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(lidar_iris PRIVATE
  ${PYTHON_LIBRARIES}
)

# include_directories(/home/lusha/miniconda3/envs/pytorch/lib/python3.7/site-packages/pybind11/include)
include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
include_directories("./LidarIris/")

add_library(LidarIris SHARED "LidarIris/LidarIris.cpp" "LidarIris/LidarIris.h" "LidarIris/fftm/fftm.cpp" "LidarIris/fftm/fftm.hpp")
add_executable(demo "demo.cpp" "LidarIris/LidarIris.cpp" "LidarIris/LidarIris.h" "LidarIris/fftm/fftm.cpp" "LidarIris/fftm/fftm.hpp")
target_link_libraries(demo ${PCL_LIBRARIES} ${OpenCV_LIBS})
