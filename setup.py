import sys
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class BuildExt(build_ext):
    def build_extensions(self):
        compiler_type = self.compiler.compiler_type
        for ext in self.extensions:
            if compiler_type == "msvc":
                ext.extra_compile_args = [
                    "/O2",
                    "/std:c++17",
                    "/EHsc",
                    "/DVTF_BUILDING_DLL",
                ]
            else:
                ext.extra_compile_args = [
                    "-O3",
                    "-std=c++17",
                    "-fPIC",
                    "-funroll-loops",
                    "-DVTF_BUILDING_DLL",
                ]
                if sys.platform != "darwin":
                    ext.extra_link_args = ["-ldl"]
        super().build_extensions()

vtf_extension = Extension(
    "_libvtf",
    sources=["vtf_core.cpp"],
    include_dirs=["."],
    language="c++",
)

readme_file = Path(__file__).resolve().parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="venkey-tick-format",
    version="1.0.0",
    description="VTF (Venkey Tick Format) v1.0 — High-Frequency Option Market Data Format & Zero-Copy Engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="venkey2k",
    author_email="venkey2kkid@gmail.com",
    url="https://github.com/venkey2k/VTF",
    project_urls={
        "Source": "https://github.com/venkey2k/VTF",
        "Bug Tracker": "https://github.com/venkey2k/VTF/issues",
    },
    license="MIT",
    py_modules=["vtf"],
    ext_modules=[vtf_extension],
    cmdclass={"build_ext": BuildExt},
    package_data={"": ["libvtf.so", "vtf_core.cpp", "vtf.hpp"]},
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=["numpy>=1.20"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: C++",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
    ],
)
