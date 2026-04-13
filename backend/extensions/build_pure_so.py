import os
import subprocess
import shutil

# 1. Resolve nvcc
nvcc_path = "nvcc"
try:
    from torch.utils.cpp_extension import CUDA_HOME
    if CUDA_HOME is not None:
        nvcc_path = os.path.join(CUDA_HOME, "bin", "nvcc")
except ImportError:
    pass

if not shutil.which(nvcc_path):
    print(f"Warning: {nvcc_path} not found in PATH. Ensure CUDA is installed.")

# 2. Resolve GPU architectures
arch_flags = ""
try:
    import torch
    arch_list = torch.cuda.get_arch_list()
    # E.g. ['sm_50', ..., 'sm_75', 'sm_80', 'sm_86', 'sm_89', 'sm_90']
    valid_archs = [a for a in arch_list if a != 'sm_37']
    flags = []
    for a in valid_archs:
        # e.g., 'sm_86' -> 'compute_86'
        compute = "compute_" + a.split("_")[1]
        flags.append(f"-gencode arch={compute},code={a}")
    arch_flags = " ".join(flags)
except Exception as e:
    print(f"Could not dynamically determine GPU architectures via PyTorch: {e}")
    # Fallback to broadly compatible modern archs
    arch_flags = "-gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89"

cmds = [
    f"{nvcc_path} -shared -o libtricked_ops.so extract_feature.cu -O3 -Xcompiler -fPIC -lcudart {arch_flags}"
]

for cmd in cmds:
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)