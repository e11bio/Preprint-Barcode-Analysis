# Comprehensive Analysis of AAV Infection Dynamics and Neuron Populations

## 1. Epitope Co-occurrence Analysis
This analysis reveals which epitopes tend to be expressed together in each population.
The heatmaps show co-expression frequencies, with the differential map highlighting
epitope combinations that are enriched in the Overexpressors population.

## 2. Hierarchical Clustering Analysis
This dendrogram shows how neurons cluster based on their epitope expression patterns,
revealing potential subpopulations within the main Expected Population and Overexpressors groups.

## 3. Infection Marker Correlation Analysis
This analysis examines the relationship between viral load (as measured by total signal intensity)
and the number of channels expressed, providing insights into infection dynamics in each population.

## 4. Channel Preference Analysis
This analysis identifies epitopes that are preferentially expressed in one population versus the other,
potentially revealing biological differences in infection susceptibility or expression mechanisms.


## 7. NGS Correlation Analysis
This analysis correlates epitope expression frequencies with viral and plasmid frequencies from NGS data,
investigating how the input viral pool composition influences expression patterns in each population.

## 8. Hamming Distance Analysis
This analysis examines the Hamming distances between barcodes, focusing on each neuron's closest match.
The mean Hamming distance to the closest match is 0.71 bits.
The distribution of these distances provides insights into the uniqueness and distinguishability of the barcodes.

## 9. Barcode Collision Analysis
This analysis quantifies how many cells have unique barcodes versus how many share the same barcode pattern with other cells.
Out of 1183 total cells, there are 685 unique barcode patterns.
### Collision Distribution Table

| Collisions | Number of Cells | Percentage |
|:----------:|:-------------:|:----------:|
| 0.0 | 510.0 | 43.11% |
| 1.0 | 158.0 | 13.36% |
| 2.0 | 141.0 | 11.92% |
| 3.0 | 76.0 | 6.42% |
| 4.0 | 70.0 | 5.92% |
| 5.0 | 36.0 | 3.04% |
| 6.0 | 14.0 | 1.18% |
| 7.0 | 16.0 | 1.35% |
| 8.0 | 9.0 | 0.76% |
| 23.0 | 24.0 | 2.03% |
| 24.0 | 25.0 | 2.11% |
| 28.0 | 29.0 | 2.45% |
| 29.0 | 30.0 | 2.54% |
| 44.0 | 45.0 | 3.80% |

Notes:
- Largest collision of 44 is actually just all segments that express all zeros
- second largest group of collisions are segments tthat only express E1 Barcode
- the group of 28 collisions is the 29 cells that express 18 channels. 
- following 25 and 24 collision are also S1 and ALFA only expressing segments. 

