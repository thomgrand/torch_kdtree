# -*- coding: utf-8 -*-
#This script is based on
#https://github.com/pybind/cmake_example
import os
import re
import subprocess
import sys
import pathlib
from shutil import copyfile

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    #"win-arm32": "ARM",
    #"win-arm64": "ARM64",
}


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):

    #def run(self):
    #    print("Running build...")
    #    for ext in self.extensions:
    #        self.build_extension(ext)

    #    super().run()

    def build_extension(self, ext):
        print("Building extension...")
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cfg = "Debug" if self.debug else "Release"

        #https://stackoverflow.com/questions/42585210/extending-setuptools-extension-to-use-cmake-in-setup-py
        # these dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        #build_temp = pathlib.Path(self.build_temp)
        #build_temp.mkdir(parents=True, exist_ok=True)
        #extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        #extdir.mkdir(parents=True, exist_ok=True)
        extdir_path = pathlib.Path(self.get_ext_fullpath(ext.name))


        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(str(extdir_path.parent.absolute())), #self.build_temp), #extdir),
            "-DPYTHON_EXECUTABLE={}".format(sys.executable),
            "-DEXAMPLE_VERSION_INFO={}".format(self.distribution.get_version()),
            "-DCMAKE_BUILD_TYPE={}".format(cfg),  # not used on MSVC, but no harm
        ]
        build_args = []

        """
        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator:
                try:
                    import ninja  # noqa: F401

                    cmake_args += ["-GNinja"]
                except ImportError:
                    pass

        else:
        """
        # Single config generators are handled "normally"
        single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

        # CMake allows an arch-in-generator style for backward compatibility
        contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

        # Specify the arch if using MSVC generator, but only if it doesn't
        # contain a backward-compatibility arch spec already in the
        # generator name.
        if not single_config and not contains_arch:
            cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

        # Multi-config generators have a different way to specify configs
        if not single_config:
            #cmake_args += [
            #    "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
            #]
            build_args += ["--config", cfg]

        print("Building extension #2...")

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += ["-j{}".format(self.parallel)]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
            
        print("Build temp dir: ", self.build_temp, extdir)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )

        #copyfile(src, dst)

        print("Args: ", ext.sourcedir, cmake_args, build_args)


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="cp_kdtree",
    version="1.0",
    author="Thomas Grandits",
    author_email="tomdev@gmx.net",
    description="Implementation of a tf_kdtree in Cupy",
    packages=["cp_kdtree"],
    long_description="",
    ext_modules=[CMakeExtension("cp_knn")],
    install_requires=["numpy>=1.20"],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    extras_require={
            "gpu": ["cupy"],
            "test": ["pytest", "scipy"]
        },
    package_data={
        "cp_kdtree": ["*.py", "*.pyd", "*.so"],
    }
)