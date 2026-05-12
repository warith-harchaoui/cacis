from setuptools import setup, find_packages
from pathlib import Path

this_dir = Path(__file__).parent
readme_path = this_dir / "README.md"
# README is not always present in deployment images. Fall back to a one-liner so
# `pip install -e .` works regardless.
readme = readme_path.read_text(encoding="utf-8") if readme_path.exists() else (
    "Cost-aware classification losses (Fenchel–Young / Sinkhorn) + fraud benchmark"
)

setup(
    name="cost-aware-losses",
    version="0.1.0",
    description="Cost-aware classification losses (Fenchel–Young / Sinkhorn) + fraud benchmark",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Warith Harchaoui",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "numpy>=1.20",
        "scipy>=1.7",
    ],
    extras_require={
        "examples": [
            "pandas>=1.5",
            "scikit-learn>=1.2",
            "matplotlib>=3.6",
            "tqdm>=4.65",
        ],
    },
)
