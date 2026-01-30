from setuptools import setup, find_packages

setup(
    name="mamba-forecast",
    version="0.1.0",
    description="Mamba-based time series forecasting for Polymarket and other 1D sequence data.",
    author="Your Name",
    author_email="you@example.com",
    url="https://github.com/your-username/your-repo-name",
    packages=find_packages(exclude=("tests", "examples", "notebooks")),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
        "pandas>=1.3.0",
        "torch>=2.0.0",
        "mamba-ssm>=2.2.0",
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "mamba",
        "state-space model",
        "time series forecasting",
        "polymarket",
        "machine learning",
        "pytorch",
    ],
)
