from setuptools import setup, find_packages

setup(
    name="online-synthetic-correspondence",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "pyyaml",
        # Add other dependencies as needed
    ],
)
