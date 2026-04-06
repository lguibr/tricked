import os
import subprocess

cmds = [
    "nvcc -shared -o ../tricked_ops.so extract_feature.cu -O3 -Xcompiler -fPIC -lcudart"
]

for cmd in cmds:
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)
