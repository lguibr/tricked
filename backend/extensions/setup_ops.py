import subprocess
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import shutil

setup(
    name="tricked_ops",
    ext_modules=[
        CUDAExtension(
            "libtricked_ops",
            [
                "custom_ops.cpp",
                "extract_feature.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
