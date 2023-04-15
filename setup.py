from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

functions_module = Extension(
    name='LidarIris',
    sources=['LidarIris.cpp'],
    include_dirs=['/usr/local/cuda-11.0/include'],
    library_dirs=['./LiDAR-Iris/build'],
    libraries=['lidar_iris'],
    language='c++',
    extra_compile_args=['-std=c++11', '-O3', '-fPIC', '-fopenmp'],
    extra_link_args=['-std=c++11', '-O3', '-fPIC', '-fopenmp']
)

setup(ext_modules=[functions_module])
