#!/bin/bash
set -e

# Move to the docs directory
cd "$(dirname "$0")"

# Create directories if they don't exist
mkdir -p source/api

# Instead of removing everything, let's create the critical files manually first
echo "Creating core documentation files..."

# Create api/index.rst file
cp templates/index.rst source/api/

# Create modules.rst with CORRECT references
cp templates/modules.rst source/api/

# Create verus.rst file with minimal content - let generate_api_docs.py handle subpackages
cp templates/verus.rst source/api/

# Now run the API doc generator to create the rest of the files
echo "Generating API documentation..."
python generate_api_docs.py

# Convert Markdown to RST if needed
echo "Converting Markdown files to RST if needed..."
if command -v pandoc &> /dev/null; then
    for mdfile in source/user_guide/*.md; do
        if [ -f "$mdfile" ]; then
            rstfile="${mdfile%.md}.rst"
            echo "Converting $mdfile to $rstfile"
            pandoc "$mdfile" -f markdown -t rst -o "$rstfile"
        fi
    done
else
    echo "Warning: pandoc not installed. Cannot convert Markdown to RST."
fi

# Fix the indentation issue in the with_timeout docstring
echo "Building Sphinx documentation..."
make clean
make html

echo "Documentation built successfully!"
echo "Open ./build/html/index.html in your browser to view it."