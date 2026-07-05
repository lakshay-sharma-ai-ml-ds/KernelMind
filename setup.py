from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="kernelmind",
    version="0.1.0",
    author="ML Systems Engineer",
    description="An agentic ML compiler and GPU kernel optimizer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "pydantic>=2.0.0",
        "psutil>=5.9.0",
        "py-cpuinfo>=9.0.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
        "tensorboard>=2.12.0",
        "anthropic>=0.7.0",
        "python-dotenv>=1.0.0",
    ],
)
