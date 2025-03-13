"""Generate API documentation for VERUS."""

import os
import sys
from collections import defaultdict
from pathlib import Path


def find_modules_and_packages(package_name, package_path):
    """Find all modules and subpackages in a package."""
    modules = []
    packages = []
    package_contents = defaultdict(list)

    # Ensure the package directory exists
    if not os.path.isdir(package_path):
        return modules, packages, package_contents

    # Find modules and subpackages
    for item in os.listdir(package_path):
        item_path = os.path.join(package_path, item)

        # Skip __pycache__ and hidden directories
        if item.startswith("__") or item.startswith("."):
            continue

        # If it's a Python file
        if item.endswith(".py") and item != "__init__.py":
            module_name = item[:-3]  # Remove .py extension
            full_module_name = f"{package_name}.{module_name}"
            modules.append(full_module_name)

            # Add to parent package contents
            package_contents[package_name].append(full_module_name)

        # If it's a directory with __init__.py (a subpackage)
        elif os.path.isdir(item_path) and os.path.isfile(
            os.path.join(item_path, "__init__.py")
        ):
            subpackage_name = f"{package_name}.{item}"
            packages.append(subpackage_name)

            # Add to parent package contents
            package_contents[package_name].append(subpackage_name)

            # Recursively find modules in the subpackage
            sub_modules, sub_packages, sub_contents = find_modules_and_packages(
                subpackage_name, item_path
            )
            modules.extend(sub_modules)
            packages.extend(sub_packages)

            # Merge sub_contents into package_contents
            for pkg, contents in sub_contents.items():
                package_contents[pkg].extend(contents)

    return modules, packages, package_contents


def create_module_file(module_name, output_dir):
    """Create RST file for a module."""
    filename = f"{module_name}.rst"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        f.write(f"{module_name} module\n")
        f.write("=" * len(f"{module_name} module") + "\n\n")
        f.write(f".. automodule:: {module_name}\n")
        f.write("   :members:\n")
        f.write("   :undoc-members:\n")
        f.write("   :show-inheritance:\n")
        f.write("   :noindex:\n")

    print(f"Created module documentation: {filepath}")


def create_package_file(package_name, contents, output_dir):
    """Create RST file for a package with toctree for submodules."""
    filename = f"{package_name}.rst"
    filepath = os.path.join(output_dir, filename)

    # Don't overwrite manually created package files
    if package_name == "verus" and os.path.exists(filepath):
        print(f"Skipping existing package file: {filepath}")
        return

    # Sort contents to have subpackages first, then modules
    subpackages = [
        item for item in contents if item.count(".") > package_name.count(".") + 1
    ]
    modules = [item for item in contents if item not in subpackages]

    with open(filepath, "w") as f:
        f.write(f"{package_name} package\n")
        f.write("=" * len(f"{package_name} package") + "\n\n")
        f.write(f".. automodule:: {package_name}\n")
        f.write("   :members:\n")
        f.write("   :show-inheritance:\n")

        # For main package, add :noindex:
        if package_name == "verus":
            f.write("   :noindex:\n")

        # Add subpackages toctree if any
        if subpackages:
            f.write("\nSubpackages\n")
            f.write("-----------\n\n")
            f.write(".. toctree::\n")
            f.write("   :maxdepth: 1\n\n")
            for subpackage in sorted(subpackages):
                f.write(f"   {subpackage}\n")

        # Add submodules toctree if any
        if modules:
            f.write("\nSubmodules\n")
            f.write("----------\n\n")
            f.write(".. toctree::\n")
            f.write("   :maxdepth: 1\n\n")
            for module in sorted(modules):
                f.write(f"   {module}\n")

    print(f"Created package documentation: {filepath}")


def main():
    # Setup paths
    docs_dir = Path(__file__).parent
    src_dir = docs_dir.parent / "src"
    output_dir = docs_dir / "source" / "api"

    # Add src directory to Python path
    sys.path.insert(0, str(src_dir))

    # Find all modules and packages in the verus package
    modules, packages, package_contents = find_modules_and_packages(
        "verus", os.path.join(src_dir, "verus")
    )

    # Create RST files for each module
    for module in modules:
        create_module_file(module, output_dir)

    # Create RST files for each package with proper toctrees
    for package in sorted(packages):
        contents = package_contents[package]
        create_package_file(package, contents, output_dir)

    print(
        f"Generated documentation for {len(modules)} modules and {len(packages)} packages"
    )


if __name__ == "__main__":
    main()
