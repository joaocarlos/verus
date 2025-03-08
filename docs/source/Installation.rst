# Installation

## Prerequisites

-   Python 3.8 or higher
-   pip (Python package manager)
-   Spatial libraries dependencies (GDAL, Proj4)

## Basic Installation

.. code-block:: bash

    pip install verus

## Development Installation

.. code-block:: bash

    git clone https://github.com/yourusername/verus.git
    cd verus
    pip install -e ".[dev]"

## Optional Dependencies

For visualization tools:

.. code-block:: bash

    pip install -e ".[viz]"

For documentation building:

.. code-block:: bash

    pip install -e ".[docs]"
