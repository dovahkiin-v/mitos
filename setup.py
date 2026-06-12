"""Setup script for Mitos package installation."""

from setuptools import setup, find_packages

setup(
    name="mitos-adr",
    version="0.1.0",
    packages=find_packages(),
    # Ship the canonical format spec INSIDE the package — `mitos init` /
    # load_format_spec() read `mitos/format-spec.md` from the installed package
    # dir, so it must be bundled in the wheel (an editable install reads it from
    # the source tree, which silently hid this gap until a real `pip install`).
    include_package_data=True,
    package_data={"mitos": ["format-spec.md"]},
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
