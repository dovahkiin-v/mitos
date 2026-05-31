"""Setup script for Mitos package installation."""

from setuptools import setup, find_packages

setup(
    name="mitos",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "filelock>=3.0.0",
        "google-genai>=1.66.0",
        "anthropic>=0.84.0",
        "mcp>=1.26.0",
        "requests>=2.0.0",
    ],
    entry_points={
        "console_scripts": [
            "mitos=mitos.cli:main",
        ],
    },
    python_requires=">=3.8",
)
