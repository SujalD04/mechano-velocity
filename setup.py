"""
Mechano-Velocity Package Setup

Install locally with: pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="mechano-velocity",
    version="0.1.0",
    author="Mechano-Velocity Team",
    description="Physics-Informed Graph Neural Network for Spatial Transcriptomics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/mechano-velocity",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.4.0",
        "scanpy>=1.9.0",
        "anndata>=0.8.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "velocity": ["scvelo>=0.2.4"],
        "spatial": ["squidpy>=1.2.0"],
        "gnn": ["torch>=2.0.0", "torch-geometric>=2.0.0"],
        "full": [
            "scvelo>=0.2.4",
            "squidpy>=1.2.0",
            "torch>=2.0.0",
            "torch-geometric>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mechano-velocity=mechano_velocity.cli:main",
        ],
    },
)
