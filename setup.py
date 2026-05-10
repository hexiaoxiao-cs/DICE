"""
Setup script for DICE package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dice-inversion",
    version="1.0.0",
    author="DICE Authors",
    author_email="your.email@example.com",
    description="Discrete Inversion for Controllable Editing of Masked Generative Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/DICE",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/DICE/issues",
        "Documentation": "https://github.com/yourusername/DICE#readme",
        "Paper": "https://arxiv.org",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    package_dir={"": "."},
    packages=find_packages(where="."),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "Pillow>=9.5.0",
        "tqdm>=4.65.0",
        "transformers>=4.30.0",
        "open_clip_torch>=2.20.0",
        "matplotlib>=3.7.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "evaluation": [
            "lpips>=0.1.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "dice-edit=scripts.image_editing:main",
            "dice-reconstruct=scripts.image_reconstruction:main",
            "dice-eval=evaluation.eval_editing:main",
        ],
    },
)
