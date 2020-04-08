#!/usr/bin/env python3

from typing import List, Dict
from math import sqrt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np

def cluster_sentences_dbscan(vectors: List[List], pca_dim: int = 10, eps: float = 3, min_samples: int = 5, metric: str = 'euclidean', remove_noise_label: bool = True) -> Dict[str, List[List]]:
    """
    Returns a clustering of the sentence embeddings
    Args:
      vectors (List[List]): Sentence Embeddings.
      pca_dim (int): Number of dimensions for the PCA.
      eps (float) : Epsilon to use for the DBSCAN
      min_samples (int) : Min number of samples to have before considering a core point (DBSCAN)
      metric (str) : Metric to use for the DBSCAN (ex: 'cosine' or 'euclidean')
      remove_noise_label (bool) : remove the -1 cluster of the dbscan which contains noisy data
    Returns:
    clustering  : Dict wich contains for each cluster label the list of vectors
    """
    scaled = StandardScaler().fit_transform(vectors)
    pca_dim = pca_dim
    pca = PCA(n_components=pca_dim)
    pca.fit(scaled)
    # We take all "pca_dim" dimensions or less if cumulative explained variance > 0.9
    if np.sum(pca.explained_variance_ratio_) > 0.90:
        for i in range(pca_dim - 1, 2, -1):
            if np.sum(pca.explained_variance_ratio_[0:i]) < 0.90:
                pca_dim = i + 1
                break
    pcad = pca.transform(scaled)
    pcad = pcad[:, 0:pca_dim]
    dbscan = DBSCAN(metric=metric, eps=eps, min_samples=min_samples)
    dbscan.fit(pcad)
    labeled = list(zip(vectors, dbscan.labels_))
    clustering = dict()
    for el in labeled:
        if remove_noise_label:
            if el[1] == -1:
                continue
        key = str(el[1])
        if key in clustering:
            clustering[key].append(list(el[0]))
        else:
            clustering[key] = [list(el[0])]
    return clustering


def nearest_to_centroid(cluster: List[List], k: int) -> List[List]:
    """
    Returns the k most nearest sentences from the centroid of a cluster

    Args:
        cluster (List[List]): List of vectors of the cluster sentences.
        K (int): Number of vectors to keep.
    Returns:
      nearest_vec  : [K nearest vector from the centroid of the cluster]
    """
    sum_element = [0] * len(cluster[0])
    for vec in cluster:
        sum_element = [sum(x) for x in zip(sum_element, vec)]
    centroid = [x / len(cluster) for x in sum_element]
    dist = {}
    index = 0
    for vec in cluster:
        dist[index] = sqrt(sum([(a - b)**2 for a, b in zip(centroid, vec)]))
        index += 1
    dist = {k: v for k, v in sorted(dist.items(), key=lambda item: item[1])}
    nearest_vec = [cluster[i] for i in list(dist.keys())[0:k]]
    return nearest_vec


def average_answers(clusters: Dict[str, List[List]],
                   k: int) -> Dict[str, List[List]]:
    """
    Returns for each cluster the the k most nearest sentences from its centroid

    Args:
        clusters (Dict[str, List[List]]): dict of clusters.
        K (int): Number of vectors to keep.
    Returns:
        average_sentences: dict with cluster in key and list of the k nearest vector from the centroid as value
    """
    average_sentences = {}
    for cl in clusters.keys():
        average_sentences[cl] = nearest_to_centroid(clusters[cl], k)

    return average_sentences
