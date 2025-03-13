# Documentation Build System

This guide explains how to use and adapt the template-based documentation build system for future projects.

## Overview

This documentation system uses a combination of:

-   Template RST files for core documentation structure
-   Automatic discovery of modules and packages
-   Python-generated API documentation
-   Sphinx for building the final HTML documentation

## Prerequisites

-   **Python 3.x** with Sphinx installed: `pip install sphinx sphinx_rtd_theme autodoc`
-   **Bash shell** for running the build script
-   **Pandoc** (optional) for converting Markdown to RST: `brew install pandoc` (macOS) or `apt install pandoc` (Linux)

## Directory Structure

```
project/
│
├── src/                  # Source code
│   └── package_name/     # Your Python package
├── docs/
│   ├── build/            # Generated HTML docs
│   ├── source/           # Documentation source files
│   │   ├── api/          # API documentation (generated)
│   │   ├── examples/     # Examples and tutorials
│   │   └── user_guide/   # User guide content
│   ├── templates/        # Core template files
│   │   ├── index.rst     # API index template
│   │   ├── modules.rst   # Modules listing template
│   │   └── verus.rst     # Main package template
│   ├── build_docs.sh     # Documentation build script
│   └── generate_api_docs.py  # API documentation generator
```

## Template Structure and Purpose

### index.rst

This template creates the API documentation landing page and establishes the high-level navigation structure:

```rst
API Reference
=============

Complete reference documentation for your package.

.. toctree::
   :maxdepth: 1

   modules
   verus
```

### modules.rst

Provides a general module overview that includes the main package:

```rst
API Reference
=============

.. toctree::
   :maxdepth: 2

   verus
```

### verus.rst (Main Package Template)

Defines the structure for documenting your main package:

```rst
verus package
=============

.. automodule:: verus
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Subpackages
-----------

.. toctree::
   :maxdepth: 1

   verus.clustering
   verus.data
   verus.grid
   verus.utils

Submodules
----------

.. toctree::
   :maxdepth: 1

   verus.verus
```

This creates a hierarchical documentation structure where:

-   The package itself is documented at the top
-   Subpackages are listed and linked in the middle
-   Individual modules are listed and linked at the bottom

## Documentation Hierarchy

The templates establish a clear hierarchy for your API documentation:

1. **API Index** (index.rst) - The landing page
2. **Package Overview** (verus.rst) - Documents the main package and lists subpackages
3. **Subpackage Pages** (auto-generated) - Document each subpackage and list their modules
4. **Module Pages** (auto-generated) - Document individual Python modules with all classes and functions

This hierarchy makes it easy for users to navigate from general to specific content.

## How to Use

1. **Copy the build system**: Copy the `docs` directory to your project
2. **Update templates**: Edit files in `templates` dir to match your project name
3. **Replace placeholders**: Use `{{MODULES_LIST}}` to auto-populate package structure
4. **Run the build script**:

    ```bash
    cd docs
    ./build_docs.sh
    ```

## Customization for New Projects

1. **Change package name**: Replace "verus" with your package name in:

    - build_docs.sh
    - templates/\*.rst
    - generate_api_docs.py

2. **Add custom templates**: Create additional template files in `templates/`

3. **Extend module discovery**: Modify the `discover_modules()` function if needed

## Troubleshooting

1. **Missing modules**: If modules aren't discovered, check that:

    - `src_dir` path is correct in build_docs.sh
    - Modules have proper `__init__.py` files

2. **Docstring warnings**: Common docstring issues are automatically fixed, but you may need to manually fix more complex issues

3. **Template errors**: Ensure your templates have correct RST syntax and all placeholders are properly formatted

## How It Works

1. The build script discovers modules in your package
2. Templates are populated with the discovered module names
3. RST files are generated in the source/api directory
4. Sphinx builds HTML documentation from these files

## Tips for Future Projects

-   Keep templates organized by topic
-   Use consistent naming conventions
-   Add comments in templates to explain structure
-   Consider adding a VERSION placeholder for automatic versioning

By moving templates to separate files and using automatic module discovery, this system is much simpler to maintain and adapt to new projects.
