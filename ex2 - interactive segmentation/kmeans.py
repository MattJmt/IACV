import numpy as np


def kmeans_fit(data, k, n_iter=500, tol=1.e-4):
    """
    Fit kmeans
    
    Args:
        data ... Array of shape n_samples x n_features
        k    ... Number of clusters
        
    Returns:
        centers   ... Cluster centers. Array of shape k x n_features
    """
    N, P = data.shape
    
    # Create a random number generator
    # Use this to avoid fluctuation in k-means performance due to initialisation
    rng = np.random.default_rng(6174)
    
    # Initialise clusters
    centroids = data[rng.choice(N, k, replace=False)]
    
    # Iterate the k-means update steps
    #
    # TO IMPLEMENT
    #
            
    for _ in range(n_iter):
        centroids_i = centroids.copy()
        
        # Assign data points to clusters
        distances = np.linalg.norm(data[:, None] - centroids, axis=2)
        cluster = np.argmin(distances, axis=1)
        
        # Update centroids
        for i in range(k):
            if np.sum(cluster == i) > 0:
                centroids[i] = np.mean(data[cluster == i], axis=0)

        # Check convergence
        if np.linalg.norm(centroids - centroids_i) < tol:
            break
    
    return centroids


def compute_distance(data, clusters):
    """
    Compute all distances of every sample in data, to every center in clusters.
    
    Args:
        data     ... n_samples x n_features
        clusters ... n_clusters x n_features
        
    Returns:
        distances ... n_samples x n_clusters
    """

    return -1


def kmeans_predict_idx(data, clusters):
    """
    Predict index of closest cluster for every sample
    
    Args:
        data     ... n_samples x n_features
        clusters ... n_clusters x n_features
    """
    # TO IMPLEMENT