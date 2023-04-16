cmake_minimum_required(VERSION 3.8)

SET(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_FIND_LIBRARY_PREFIXES ${CMAKE_FIND_LIBRARY_PREFIXES} "lib")
set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES} ".so")

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/lib/x86_64-linux-gnu/cmake")
set(OpenCV_DIR "/usr/local/share/OpenCV")

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PythonInterp 3)
find_package(PythonLibs 3)
find_package(pybind11 REQUIRED)

set(pybind11_INCLUDE_DIR "/home/lusha/miniconda3/envs/pytorch/lib/python3.7/site-packages/pybind11/include")

include_directories(${PYTHON_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${pybind11_INCLUDE_DIR})
link_directories(${PCL_LIBRARY_DIRS})

add_library(LidarIris SHARED "LidarIris/LidarIris.cpp" "LidarIris/LidarIris.h" "LidarIris/fftm/fftm.cpp" "LidarIris/fftm/fftm.hpp")
target_include_directories(LidarIris PRIVATE ${Python_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(LidarIris PRIVATE ${Python_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})

pybind11_add_module(lidar_iris LidarIris/LidarIris.cpp)
target_include_directories(lidar_iris PUBLIC ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_include_directories(lidar_iris PRIVATE ${Python_INCLUDE_DIRS})
target_link_libraries(lidar_iris PRIVATE ${Python_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})

# cmake_minimum_required (VERSION 3.8)

# SET(CMAKE_BUILD_TYPE Release)
# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_FIND_LIBRARY_PREFIXES ${CMAKE_FIND_LIBRARY_PREFIXES} "lib")
# set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES} ".so")

# set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/lib/x86_64-linux-gnu/cmake")
# set(OpenCV_DIR "/usr/local/share/OpenCV")

# find_package(PythonInterp 3)
# find_package(PythonLibs 3)
# set(PYTHON_INCLUDE_PATH "/home/lusha/miniconda3/envs/pytorch/bin/python3.7m")
# set(PYTHON_LIBRARIES "/home/lusha/miniconda3/envs/pytorch/lib/libpython3.7m.so")
# set(PYTHON_EXECUTABLE "/home/lusha/miniconda3/envs/pytorch/bin/python3.7m")
# set(PYTHON_INCLUDE_DIRS "/home/lusha/miniconda3/envs/pytorch/include/python3.7m")
# # set(pybind11_DIR "/home/lusha/miniconda3/envs/pytorch/share/cmake/pybind11")
# set(pybind11_DIR "/home/lusha/miniconda3/envs/pytorch/lib/python3.7/site-packages/pybind11/include")

# project (LidarIris)

# find_package(PCL REQUIRED)
# find_package(OpenCV REQUIRED)
# # find_package(Python 3 REQUIRED)
# # find_package(NumPy REQUIRED)
# # find_package(PythonLibs REQUIRED)
# # find_package(pybind11 REQUIRED)

# include_directories(${PYTHON_INCLUDE_DIRS})
# # include_directories(pybind11/include)
# include_directories(${pybind11_DIR})

# add_subdirectory(pybind11)
# pybind11_add_module(lidar_iris
# LidarIris/LidarIris.cpp
# )
# target_include_directories(lidar_iris PUBLIC
#   include
#   ${PCL_INCLUDE_DIRS}
#   ${OpenCV_INCLUDE_DIRS}
# )
# target_include_directories(lidar_iris PRIVATE 
#   include
#   ${Python_INCLUDE_DIRS}
# )
# target_link_libraries(lidar_iris PRIVATE
#   ${PYTHON_LIBRARIES}
# )

# include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
# link_directories(${PCL_LIBRARY_DIRS})
# add_definitions(${PCL_DEFINITIONS})
# include_directories("./LidarIris/")


# # add_library(LidarIris SHARED "LidarIris/LidarIris.cpp")
# # target_link_libraries(LidarIris PRIVATE pybind11::module Python::Python)
# add_library(LidarIris SHARED "LidarIris/LidarIris.cpp" "LidarIris/LidarIris.h" "LidarIris/fftm/fftm.cpp" "LidarIris/fftm/fftm.hpp")
# add_executable(demo "demo.cpp" "LidarIris/LidarIris.cpp" "LidarIris/LidarIris.h" "LidarIris/fftm/fftm.cpp" "LidarIris/fftm/fftm.hpp")
# target_link_libraries(demo ${PCL_LIBRARIES} ${OpenCV_LIBS})
# target_include_directories(LidarIris PRIVATE ${Python_INCLUDE_DIRS})
# target_link_libraries(LidarIris PRIVATE ${Python_LIBRARIES})
