Points of Temporal Influence (PoTIs)
====================================

Definition
----------

Points of Temporal Influence (PoTIs) are locations in an urban environment that attract varying concentrations of people throughout the day. In VERUS, they are represented as geospatial points with associated attributes that affect vulnerability calculation.

PoTIs are essentially Points of Interest (POIs) enhanced with temporal data that captures how their influence on urban vulnerability changes over time.

Characteristics
---------------

Each PoTI has the following key characteristics:

* **Spatial coordinates**: Latitude and longitude defining its location
* **Category**: Type of location (e.g., school, hospital, transportation hub, etc.)
* **Vulnerability Influence (VI)**: A numeric value representing the location's contribution to vulnerability
* **Cluster assignment**: Which cluster the PoTI belongs to (assigned during analysis)

Examples
---------

Common examples of PoTIs include:

* Schools (high density during school hours)
* Hospitals (consistent presence of vulnerable populations)
* Transportation hubs (variable crowding based on rush hours)
* Shopping malls (peak times during evenings and weekends)
* Industrial areas (worker presence during shifts)

Extraction
----------

VERUS includes a `DataExtractor` class that can pull PoTI data from OpenStreetMap:

.. code-block:: python

    extractor = DataExtractor(region="Porto, Portugal")
    poi_data = extractor.run()

The extractor fetches location data for various categories and processes them into a standardized format for use in vulnerability assessment.

Importance in Vulnerability Assessment
--------------------------------------

PoTIs are the foundation of the VERUS analysis because:

1. Their category and attributes influence vulnerability weighting
2. Their spatial distribution and vulnerability influence forms the basis for clustering
3. Their influence within each cluster affects vulnerability calculations
4. Their temporal patterns drive time-based scenario modeling

Each PoTI contributes to the overall vulnerability calculation based on its properties and the calculation method selected.