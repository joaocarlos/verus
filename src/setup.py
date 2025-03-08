import os

from setuptools import find_packages, setup

# Read the version from __init__.py to maintain a single source of truth
with open(os.path.join("verus", "__init__.py"), "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break
    else:
        version = "unknown"  # Default if not found

# Read long description from README
try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "*VERUS* - Vulnerability Evaluation for Resilient Urban Systems"

setup(
    name="verus",
    version=version,
    author="Verus Team",
    author_email="your.email@example.com",  # Replace with your email
    description="Clustering-based framework for assessing the vulnerability of urban areas during emergencies, indicating how populations are negatively affected based on the existing urban dynamics.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/verus",  # Replace with your repository URL
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/verus/issues",
        "Documentation": "https://verus.readthedocs.io/",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    # Core dependencies needed for basic functionality
    install_requires=[
        # Data processing
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        # Geospatial libraries
        "geopandas>=0.10.0",
        "shapely>=1.8.0",
        "pyproj>=3.0.0",
        "osmnx>=1.1.0",
        "rtree>=1.0.0",
        # Visualization
        "folium>=0.12.0",
        "matplotlib>=3.5.0",
        "plotly>=5.3.0",
        # Machine learning
        "scikit-learn>=1.0.0",
        # Utilities
        "colorama>=0.4.4",
        "tqdm>=4.62.0",
        "requests>=2.26.0",
    ],
    # Optional dependencies for different use cases
    extras_require={
        # Development dependencies
        "dev": [
            "pytest>=6.2.5",
            "black>=22.1.0",
            "isort>=5.10.1",
            "flake8>=4.0.1",
            "mypy>=0.910",
            "coverage>=6.2",
        ],
        # Documentation building
        "docs": [
            "sphinx>=4.3.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbsphinx>=0.8.8",
        ],
        # Extra visualization tools
        "viz": [
            "seaborn>=0.11.2",
            "plotly>=5.3.0",
            "ipywidgets>=7.6.5",
        ],
    },
    # Include non-Python files
    include_package_data=True,
    package_data={
        "verus": ["py.typed", "data/templates/*"],
    },
    # Entry points for command line tools
    entry_points={
        "console_scripts": [
            "verus-extract=verus.cli.extract:main",
            "verus-cluster=verus.cli.cluster:main",
        ],
    },
    # Zip-safe flag for egg installations
    zip_safe=False,
)


# pip install -e ".[dev]"
# pip install -e ".[docs]"
# pip install -e ".[dev,docs,viz]"
# python -m build # Creating distribution packages
