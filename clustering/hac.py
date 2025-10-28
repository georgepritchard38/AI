import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import csv
import copy

def load_data(filepath):
    file = open(filepath)
    dict = csv.DictReader(file)
    data = []
    for i, row in enumerate(dict):
        data.append(row)

    return data

def calc_features(row):
    vector = np.array([row['child_mort'], row['exports'], row['health'],
                        row['imports'], row['income'], row['inflation'],
                        row['life_expec'], row['total_fer'], row['gdpp']], dtype = np.float64)
    return vector

def hac(features, linkage_type):
    out = np.empty(((len(features) - 1), 4))
    # initialize matrix of clusters, storing the number(index in deepFeatures) of each data point cluster
    clusters = []
    for i, feauture in enumerate(features):
        clusters.append([[i], i])
    # initialize distance matrix with distances of all points
    distanceMatrix = np.empty((len(features),len(features)))
    for i, row in enumerate(features):
        for j, column in enumerate(features):
            distanceMatrix[i][j] = abs(np.linalg.norm(features[i] - features[j]))
    #iterate through each row in out, for each loop calculate 1 row of the array, starting @ 0
    for i, element in enumerate(out):
        distance = None
        closest = None
        smaller = None
        larger = None
        #Closest cluster in each iteration are defined in closest, smaller, larger
        for j, feature in enumerate(clusters):
            for n, feature in enumerate(clusters):
                if(j == n):
                    continue
                clusterDistance = None
                pointJ = None
                pointN = None
                clusterJ = None
                clusterN = None
                #computes maximium distance between two clusters
                if(linkage_type == "complete"):
                    for jj, feature in enumerate(clusters[j][0]):
                        for nn, feature in enumerate(clusters[n][0]):
                            if(clusterDistance == None or distanceMatrix[clusters[j][0][jj]][clusters[n][0][nn]] > clusterDistance):
                                clusterDistance = distanceMatrix[clusters[j][0][jj]][clusters[n][0][nn]]
                                pointJ = jj
                                pointN = nn
                                clusterJ = clusters[j]
                                clusterN = clusters[n]
                #computes minimum distance between two clusters
                elif(linkage_type == "single"):
                    for jj, feature in enumerate(clusters[j][0]):
                        for nn, feature in enumerate(clusters[n][0]):
                            if(clusterDistance == None or distanceMatrix[clusters[j][0][jj]][clusters[n][0][nn]] < clusterDistance):
                                clusterDistance = distanceMatrix[clusters[j][0][jj]][clusters[n][0][nn]]
                                pointJ = jj
                                pointN = nn
                                clusterJ = clusters[j]
                                clusterN = clusters[n]
                distance = clusterDistance
                #Updates closest, smaller, larger whenever a smaller distance is found
                if(closest == None or distance < closest):
                    closest = distance
                    if(clusterN[1] < clusterJ[1]):
                        smaller = clusterN
                        larger = clusterJ
                    else:
                        smaller = clusterJ
                        larger = clusterN
        # join smaller and larger, input ith row of out array
        out[i][0] = smaller[1]
        out[i][1] = larger[1]
        smaller[0] = smaller[0] + larger[0]
        smaller[1] = len(features) + i
        clusters.remove(larger)
        out[i][2] = closest
        out[i][3] = len(smaller[0])
        
    return out

def fig_hac(Z, names):
    fig = plt.figure()
    dendrogram = sp.cluster.hierarchy.dendrogram(Z, labels = names,leaf_rotation = 90)
    fig.tight_layout()
    return fig

def normalize_features(features):
    array = np.array(features)
    means = np.mean(array, axis = 0)
    stdevs = np.std(array, axis = 0)
    
    newFeatures = []

    for i, feature in enumerate(features):
        newFeature = []
        for j, datum in enumerate(feature):
            newDatum = (datum - means[j]) / stdevs[j]
            newFeature.append(newDatum)
        newVector = np.array(newFeature)
        newFeatures.append(newVector)

    return newFeatures

if __name__ == "__main__":
    data = load_data("/Users/georgepritchard/Desktop/VSCode/CS540/hw4/Country-data.csv")
    features = [calc_features(row) for row in data]
    names = [row["country"] for row in data]
    features_normalized = normalize_features(features)
    np.savetxt("output.txt", features_normalized)
    n = 20
    Z = hac(features[:n], linkage_type="complete")
    fig = fig_hac(Z, names[:n])
    plt.show()
