from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cadlib",
    version="1.0.0",
    author="",
    author_email="",
    description="A library for CAD representation design automation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/automating_tool_cad_representation_design",
    packages=find_packages(),  # Remove the where="src" parameter
    # Remove package_dir since we're not using src layout
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "matplotlib",
        "trimesh",
        "pythonocc-core",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
    },
)