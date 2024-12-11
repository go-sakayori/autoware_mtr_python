import os.path as osp

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(name: str, module: str, sources: list[str], cxx_std: int = 17) -> CUDAExtension:
    """Configure build settings for CUDA extensions.

    Args:
    ----
        name (str): Name of extension
        module (str): Module path to be generated.
        sources (list[str]): List of source files.
        cxx_std (int): CXX standard version. Defaults to 17.

    Returns:
    -------
        CUDAExtension: Configured `CUDAExtension` instance.

    """
    assert cxx_std in (17, 20), f"Unexpected CXX standard: {cxx_std}"
    return CUDAExtension(
        name=f"{module}.{name}",
        sources=[osp.join(*module.split("."), src) for src in sources],
        extra_compile_args=[f"-std=c++{cxx_std}", "-v"],
    )


def build_custom_ext() -> dict:
    """Build custom C++ extensions.

    Args:
    ----
        setup_kwargs (dict): Keyword arguments for setup.

    """
    ext_modules = [
        make_cuda_ext(
            name="cuda_ops",
            module="mtr.ops",
            sources=[
                "csrc/custom_ops.cpp",
                "csrc/attention/attention_func.cpp",
                "csrc/attention/attention_value_computation_kernel.cu",
                "csrc/attention/attention_weight_computation_kernel.cu",
                "csrc/knn/knn_func.cpp",
                "csrc/knn/knn_kernel.cu",
            ],
        ),
    ]

    return {
        "ext_modules": ext_modules,
        "cmdclass": {
            "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
        },
    }


if __name__ == "__main__":
    setup(name="MTR", version="0.0.3", **build_custom_ext())
