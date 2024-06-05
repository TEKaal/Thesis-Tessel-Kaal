from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt
import numpy as np

def hierarchical_clustering(distance_matrix, max_d, max_nodes):
    """
    Perform hierarchical clustering based on a distance matrix, with constraints on maximum
    distance and maximum number of nodes per cluster.

    Parameters:
    - distance_matrix: A squareform distance matrix where D[i,j] is the distance between i and j.
    - max_d: The maximum distance to use for forming clusters.
    - max_nodes: The maximum number of nodes allowed in any cluster.

    Returns:
    - clusters: A list of clusters that respects the maximum number of nodes.
    """

    # Perform hierarchical clustering using the distance matrix
    Z = linkage(distance_matrix, method='ward')

    # Start by using the maximum distance as a threshold to form initial clusters
    clusters = fcluster(Z, max_d, criterion='distance')

    # Check if any cluster exceeds the maximum number of nodes allowed
    while any(np.bincount(clusters) > max_nodes):
        max_d *= 0.95  # Reduce max_d incrementally to tighten the clustering
        clusters = fcluster(Z, max_d, criterion='distance')

    return clusters
