Development
===========

Contributing to VERUS
---------------------

Contributions to VERUS are welcome! Here's how to contribute:

1. Fork the repository
2. Create a new branch for your feature
3. Add your code and tests
4. Ensure all tests pass
5. Submit a pull request

Code Style
----------

VERUS follows the PEP 8 style guide. Please ensure your code complies with these standards.

We recommend using tools such as:

- black for code formatting
- flake8 for linting
- isort for import sorting

Testing
-------

VERUS uses simple tests to ensure the code is working as expected. To run the tests, follow the `test folder <https://github.com/joaocarlos/verus/tree/main/tests>`_.

Documentation
-------------

Ensure that all public functions and classes are well documented. Documentation is built using Sphinx. To build the documentation:

.. code-block:: bash

    cd docs
    ./build_docs.sh

Release Process
---------------

1. Update version in `src/verus/__init__.py`
2. Update CHANGELOG.md
3. Create a new release on GitHub
4. Publish to PyPI