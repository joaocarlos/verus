"""Generate API documentation for VERUS."""

import os
import pkgutil
import sys


def create_module_file(package, module_name, outdir):
    """Create RST file for a module."""
    fullname = f"{package}.{module_name}"
    with open(os.path.join(outdir, f"{fullname}.rst"), "w") as f:
        f.write(f"{fullname} module\n")
        f.write("=" * (len(fullname) + 7) + "\n\n")
        f.write(f".. automodule:: {fullname}\n")
        f.write("   :members:\n")
        f.write("   :undoc-members:\n")
        f.write("   :show-inheritance:\n")


def create_package_file(package, outdir, submodules):
    """Create RST file for a package."""
    with open(os.path.join(outdir, f"{package}.rst"), "w") as f:
        f.write(f"{package} package\n")
        f.write("=" * (len(package) + 8) + "\n\n")
        f.write(f".. automodule:: {package}\n")
        f.write("   :members:\n")
        f.write("   :undoc-members:\n")
        f.write("   :show-inheritance:\n\n")

        f.write("Submodules\n")
        f.write("----------\n\n")
        f.write(".. toctree::\n")
        f.write("   :maxdepth: 1\n\n")

        for module_name in sorted(submodules):
            f.write(f"   {package}.{module_name}\n")


def recurse_modules(package, outdir):
    """Recursively find all modules in a package."""
    submodules = []

    # Import the package
    try:
        __import__(package)
        pkg = sys.modules[package]
    except ImportError:
        print(f"Could not import {package}")
        return submodules

    for _, modname, ispkg in pkgutil.iter_modules(pkg.__path__, package + "."):
        if ispkg:
            # It's a subpackage
            sub_submodules = recurse_modules(modname, outdir)
            create_package_file(modname, outdir, sub_submodules)
        else:
            # It's a module
            create_module_file(package, modname.split(".")[-1], outdir)
            submodules.append(modname.split(".")[-1])

    return submodules


def generate_modules_rst(outdir="docs/source/api"):
    """Generate the module RST files."""
    os.makedirs(outdir, exist_ok=True)

    # Add parent directory to path for importing
    sys.path.insert(0, os.path.abspath("../src"))

    # Create the main modules.rst file
    with open(os.path.join(outdir, "modules.rst"), "w") as f:
        f.write("API Reference\n")
        f.write("=============\n\n")
        f.write(".. toctree::\n")
        f.write("   :maxdepth: 2\n\n")
        f.write("   verus\n")

    # Create the verus package documentation
    recurse_modules("verus", outdir)


if __name__ == "__main__":
    generate_modules_rst()
