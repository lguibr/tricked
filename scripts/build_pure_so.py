import os
import subprocess
import torch
from torch.utils.cpp_extension import include_paths, library_paths

abi = "1" if torch.compiled_with_cxx11_abi() else "0"
inc_args = " ".join([f"-I{p}" for p in include_paths()])
lib_args = " ".join([f"-L{p}" for p in library_paths()])

cmds = [
    f"/usr/local/cuda-13.2/bin/nvcc -c extract_feature.cu -o extract_feature.o -O3 {inc_args} -Xcompiler -fPIC -D_GLIBCXX_USE_CXX11_ABI={abi}",
    f"g++ -c custom_ops.cpp -o custom_ops.o -O3 {inc_args} -fPIC -D_GLIBCXX_USE_CXX11_ABI={abi}",
    f"g++ -shared -o ../tricked_ops.so extract_feature.o custom_ops.o {lib_args} -L/usr/local/cuda-13.2/lib64 -lcudart -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda",
]

for cmd in cmds:
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)
