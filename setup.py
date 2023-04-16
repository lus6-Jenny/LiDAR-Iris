from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

functions_module = Extension(
    name='LidarIris',
    sources=['LidarIris.cpp'],
    include_dirs=['./pybind11/include', './LiDAR-Iris/fftm'],
    library_dirs=['./LiDAR-Iris/build'],
    libraries=['lidar_iris'],
    language='c++',
    extra_compile_args=['-std=c++11', '-O3', '-fPIC', '-fopenmp'],
    extra_link_args=['-std=c++11', '-O3', '-fPIC', '-fopenmp']
)

setup(ext_modules=[functions_module])
