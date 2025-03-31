# Hierarchical Clustering Analysis of Soma Barcodes

## Overview
This analysis presents a hierarchical clustering of 147 soma barcodes across 18 epitope channels.

## Methodology
- **Data**: Binary barcode matrix where each row represents a soma and each column represents a target channel
- **Distance Metric**: Hamming distance between barcode patterns
- **Linkage Method**: ward linkage
- **Visualization**: Binary heatmap where black indicates presence (1) and white indicates absence (0)

## Interpretation
The hierarchical clustering organizes somas with similar barcode patterns together, revealing potential groups or clusters of somas that share similar epitope expression patterns. Clusters of somas with similar patterns may indicate:

1. Somas with shared lineage
2. Somas with similar functional properties
3. Potential technical artifacts or batch effects

## Channels
The 18 epitope channels displayed from left to right are:
- E2, S1, ALFA, Ty1, HA, T7, VSVG, AU5, NWS, SunTag, ETAG, SPOT, MoonTag, HSV, ProteinC, Tag100, CMyc, OLLAS

![Hierarchical Clustering Heatmap](hierarchical_clustering.png)
