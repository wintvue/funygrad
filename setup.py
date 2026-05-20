from setuptools import setup, find_packages

setup(
    name="funygrad",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy>=1.21"],
    python_requires=">=3.10",
    author="funygrad",
    description="A tinygrad-inspired deep learning framework",
)
