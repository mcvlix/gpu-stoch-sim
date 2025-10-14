import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

build_cuda = os.environ.get("SDEGPU_BUILD_CUDA", "1") == "1" and (CUDA_HOME is not None)

ext_modules = []
cmdclass = {}

if build_cuda:
    ext_modules = [
        CUDAExtension(
            name="sdegpu._cuda._ext",
            sources=[
                "src/sdegpu/_cuda/binding.cpp",
                "src/sdegpu/_cuda/em_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ]
    cmdclass = {"build_ext": BuildExtension.with_options(use_ninja=True)}
else:
    print("⚠️  Skipping CUDA extension (set CUDA_HOME and SDEGPU_BUILD_CUDA=1 to build).")

setup(
    name="sdegpu",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
