from setuptools import setup
from setuptools_rust import Binding, RustExtension
import os

# Get the directory containing setup.py
here = os.path.abspath(os.path.dirname(__file__))
# Get the parent directory (zkml root)
rust_extension_path = os.path.dirname(here)

setup(
    name="zkml",
    version="0.1.0",
    packages=["zkml"],
    rust_extensions=[
        RustExtension(
            "zkml.zkml",
            path=os.path.join(rust_extension_path, "Cargo.toml"),
            binding=Binding.PyO3,
        )
    ],
    install_requires=["setuptools-rust>=1.5.2"],
    setup_requires=["setuptools-rust>=1.5.2"],
    include_package_data=True,
    zip_safe=False,
)
