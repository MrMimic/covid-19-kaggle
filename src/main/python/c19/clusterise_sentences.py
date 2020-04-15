#!/usr/bin/env python3

import math
import time
from math import sqrt
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min, silhouette_score
from sklearn.preprocessing import StandardScaler


def compute_best_k_silhouette(closest_sentences_df: pd.DataFrame, k_min: int,
                              k_max: int) -> int:
    """
    Utilities allowing to estimate the best K values for a KMean clustering.
    It uses the Silhouette Coefficient. It relates to a model with better-defined clusters.
    The Silhouette Coefficient is defined for each sample and is composed of two scores:
        - The mean distance between a sample and all other points in the same class.
        - The mean distance between a sample and all other points in the next nearest cluster.
    Higher silhouette coefficient means well difined clusters.

    Args:
        closest_sentences_df (pd.DataFrame): The output of the top-k sentences extraction.
        k_min (int): The minimal number of clusters.
        k_max (int): The maximal number of clusters.

    Returns:
        int: The optimal number of clusters
    """
    # The vector are extracted to optimize K
    data = closest_sentences_df.vector.to_list()
    # Silhouette score is stored for each possible K
    silhouette_score_k = {}
    for n_cluster in range(k_min, k_max):
        kmeans = KMeans(n_clusters=n_cluster, n_init=1, random_state=42).fit(data)
        label = kmeans.labels_
        sil_coeff = silhouette_score(data, label, metric='euclidean')
        silhouette_score_k[sil_coeff] = n_cluster
    # Let's split high coefficients and low coefficients
    silhouette_coeffs = [[x] for x in list(silhouette_score_k.keys())]
    silhouette_score_spliter = KMeans(n_clusters=2, n_init=1, random_state=42).fit(silhouette_coeffs)
    # Which groups represent the "highest scores" ?
    tmp_df = pd.DataFrame(zip(list(silhouette_score_k.keys()),
                              silhouette_score_spliter.labels_),
                          columns=["score", "label"])
    average_score_per_group = pd.DataFrame(
        tmp_df.groupby(['label'], as_index=False).mean())
    highest_group_label = average_score_per_group[
        average_score_per_group["score"] ==
        average_score_per_group["score"].max()]["label"].values[0]
    # Get all those labels scores and max score
    max_scores = tmp_df[tmp_df["label"] ==
                        highest_group_label]["score"].to_list()
    # Which represent X clusters
    possible_k_values = [silhouette_score_k[score] for score in max_scores]
    # Let's round up possible k values leading to the highest scores
    return math.ceil(np.mean(possible_k_values))


def perform_kmean(k_closest_sentences_df: pd.DataFrame,
                  number_of_clusters: Union[int, str],
                  k_min: int = None,
                  k_max: int = None) -> pd.DataFrame:
    """
    Add a columns "cluster" and "is_closest" to the sentence dataframe.

    Args:
        k_closest_sentences_df (pd.DataFrame): The DF to be updated.
        number_of_clusters (Union[int, str]): Number of K in the Kmean. If "auto",
        perform silhouette score to determine the best K.

    Returns:
        pd.DataFrame: Updated DF.
    """
    tic = time.time()
    if number_of_clusters == "auto":
        number_of_clusters = compute_best_k_silhouette(
            closest_sentences_df=k_closest_sentences_df,
            k_min=k_min,
            k_max=k_max)

    # Clusterise vectors
    vectors = k_closest_sentences_df["vector"].tolist()
    kmean_model = KMeans(n_clusters=number_of_clusters, n_init=1, random_state=42).fit(vectors)
    # Label clusters
    k_closest_sentences_df["cluster"] = kmean_model.labels_
    # Compute closest from barycentres and store in a boolean column
    closest, _ = pairwise_distances_argmin_min(kmean_model.cluster_centers_,
                                               vectors)
    k_closest_sentences_df["is_closest"] = [
        True if vectors.index(vector) in closest else False
        for vector in vectors
    ]

    toc = time.time()
    print(
        f"Took {round((toc-tic), 2)} seconds to clusterise {k_closest_sentences_df.shape[0]} closest sentences."
    )

    return k_closest_sentences_df


def perform_dbscan(vectors: List[List],
                   pca_dim: int = 10,
                   eps: float = 3,
                   min_samples: int = 5,
                   metric: str = 'euclidean',
                   remove_noise_label: bool = True) -> Dict[str, List[List]]:
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
