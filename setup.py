from setuptools import setup, find_namespace_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools_rust import Binding, RustExtension
import os

try:
    import torch
    # Determine arch flags
    arch_list = torch.cuda.get_arch_list()
    # Pytorch might return sm_37 but modern environments drop it
    valid_archs = [a for a in arch_list if a != 'sm_37']
except Exception:
    valid_archs = ['sm_75', 'sm_80', 'sm_86', 'sm_89', 'sm_90']

setup(
    name="tricked",
    packages=find_namespace_packages(include=["tricked", "tricked.*"]),
    package_data={"tricked": ["masks.json"]},
    rust_extensions=[
        RustExtension(
            target="tricked_engine",
            path="tricked/engine/Cargo.toml",
            binding=Binding.PyO3,
            debug=False,
        )
    ],
    ext_modules=[
        CUDAExtension(
            name="libtricked_ops",
            sources=[
                "tricked/extensions/custom_ops.cpp",
                "tricked/extensions/extract_feature.cu",
            ],
            extra_compile_args={
                # Provide optimization and arch targets
                'nvcc': ['-O3'] + [f'-gencode=arch=compute_{a.split("_")[1]},code={a}' for a in valid_archs]
            }
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
