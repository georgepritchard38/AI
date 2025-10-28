## Clustering

This project uses Hierarchical Agglomerative Clustering (HAC) to cluster countries based on demographic data in the hac.py program file. Clusters can be determined using either single or complete linkage. Outputs (output.txt) an N x 4 matrix that describes the clustering process linearly. Each row describes the conjoinment of two clusters. Column 1 specifies the smaller index of the two clusters being joined. Cloumn 2 specifies the larger index of the two clusters being joined. Column 3 specifies the linkage distance between the two aforementioned clusters. Column 4 specifies the total number of countries in the new cluster. The file first_20.png visualizes the clustering of the first 20 (alphabetically) countries in country-data.csv. The world.png file visualizes the complete linkage of 100 randomly selected countries based on the provided demographic data.

## Provided vy instructors:

country demographics dataset (Country-data.csv)

function.py (visualizes the country clusters on world map)

## Personal contributions:

hac.py

first_20.png

world.png
