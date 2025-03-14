![LES2 Logo](./src/figures/les2banner.png)

# VERUS

_Vulnerability Evaluation for Resilient Urban Systems_

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter)
![OSMnx](https://img.shields.io/badge/OSMnx-1.1.1-blue?style=for-the-badge&logo=openstreetmap)
![Folium](https://img.shields.io/badge/Folium-0.12.1-green?style=for-the-badge&logo=folium)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen?style=for-the-badge)](https://les2feup.github.io/verus/)

## Description

VERUS (Vulnerability Evaluation for Resilient Urban Systems) is a Python library for extracting points of interest from OpenStreetMap, clustering them based on spatial proximity and time-based vulnerability indices, and analyzing urban vulnerability patterns.

## Documentation

Comprehensive documentation is available at [https://les2feup.github.io/verus/](https://les2feup.github.io/verus/)

## Publications

-   Bittencourt, J. C. N., Costa, D. G., Portugal, P., & Vasques, F. (2024). A data-driven clustering approach for assessing spatiotemporal vulnerability to urban emergencies. Sustainable Cities and Society, 108, 105477. <https://doi.org/10.1016/j.scs.2024.105477>

## Installation

```bash
pip install verus
```

## Reproducing Results

To reproduce the results from the latest article, run the following Jupyter Notebooks:

-   [Porto, Portugal](./notebooks/experiments/01-Porto.ipynb)
-   [Lisbon, Portugal](./notebooks/experiments/02-Lisbon.ipynb)
-   [Paris, France](./notebooks/experiments/03-Paris.ipynb)

## Basic Usage

```python
from verus import VERUS
from verus.data import DataExtractor, TimeWindowGenerator
from verus.grid import HexagonGridGenerator

# Extract default urban data from OpenStreetMap
extractor = DataExtractor(region="Porto, Portugal")
poi_data = extractor.run()

# Define default time-based vulnerability scenarios
twg = TimeWindowGenerator(reference_date="2023-11-06")
time_windows = twg.generate_from_schedule()

# Generate analysis grid
grid = HexagonGridGenerator(region="Porto, Portugal", edge_length=250)
hex_grid = grid.run()

# Run vulnerability assessment
assessor = VERUS(place_name="Porto")

# Load extracted data
assessor.load(
    potis_df=poi_data,
    time_windows_dict=time_windows,
    zones_gdf=hex_grid,
)

# Perform assessment and get results
evaluation_time = tw_gen.to_unix_epoch("2023-11-06 17:30:00")
results = assessor.run(evaluation_time=evaluation_time)

# Visualize results
map_obj = assessor.visualize()

# Save results to file
assessor.save("./results/porto/")
```

## Features

-   Extract points of interest from OpenStreetMap for any city
-   Create hexagonal grid systems for spatial analysis
-   Implement OPTICS density-based clustering for identifying urban patterns
-   Apply K-means clustering with Haversine distance for geographic data
-   Generate time-dependent vulnerability assessments
-   Create interactive maps for visualization with folium
-   Export results in various formats (GeoJSON, CSV, HTML)

## Contributions

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a pull request

Please make sure to update tests and documentation as appropriate.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

-   Research developed at the Laboratory of Emergent Smart Systems (LES2) at the Faculty of Engineering of University of Porto
-   This work was supported by the Associate Laboratory Advanced Production and Intelligent Systems – ARISE LA/P/0112/2020 (DOI 10.54499/LA/P/0112/2020), by the Base Funding (UIDB/00147/2020) and Programmatic Funding (UIDP/00147/2020) of the R\&D Unit Center for Systems and Technologies -- SYSTEC, and by the Fundação para a Ciência e a Tecnologia (FCT) through the PhD scholarship (2024.02446.BD).
