#!/usr/bin/env python3
"""
Parameters for the Kaggle Covid19 project.
Default value can be erased as follow:

    >>> param = parameters.Parameters(query=parameters.Query(k_min=3))
    >>> param.query
    >>> Query(top_k_sentences=50, k_min=3, k_max=10)
"""

from dataclasses import dataclass
import os
from typing import Union


@dataclass
class Embedding:

    local_path: str = os.path.join("global_df_w2v_tfidf.parquet")
    """
    str: The path to load the pre-trained vectors. If not found, will be downloaded.
    """

    dimension: int = 100
    """
    int: The number of dimension of the pre-trained vectors.
    """

    word_aggregation_method: str = "mowe"
    """
    str: The word aggregation method to get sentences vectors (`mowe` or `sowe`)
    """

    weight_with_tfidf: bool = True
    """
    bool: Either each word vector should be weighted with TF-IDF during sentence vector computing.
    """


@dataclass
class Database:

    local_path: str = os.path.join(f"database.sqlite")
    """
    str: Local path of the database.
    """

    kaggle_data_path: str = os.path.join("kaggle_data")
    """
    str: Local path of the kaggle data.
    """

    only_newest: bool = False
    """
    bool: If True, will drop all article published before 2019 to get a light SQL DB.
    """

    only_covid: bool = False
    """
    bool: If True, will drop all article with abstract wich do not contain a covid-19 synonym.
    """


@dataclass
class PreProcessing:

    stem_words: bool = False
    """
    bool: Should stem words during pre-processing.
    """

    remove_numeric: bool = True
    """
    bool: Should remove numeric tokens during pre-processing.
    """

    batch_size: int = 1000
    """
    int: The number of close sentences to be tajen in account when computing clustering.
    """

    max_body_sentences: int = 10
    """
    int: The number of close sentences to be tajen in account when computing clustering. 0 means all.
    """


@dataclass
class Query:

    top_k_sentences_number: int = 500
    """
    int: The number of close sentences to be taken in account when computing clustering.
    """

    top_k_sentences_distance: float = 0.8
    """
    float: The score threshold to filter sentences having distance < arg.
    """

    filtering_method: str = "distance"
    """
    str: The filter method to be used for clustering ('distance' or 'number').
    """

    k_min: int = 2
    """
    int: The minimal number of clusters of sentence to compute with Silhouette score.
    """

    k_max: int = 10
    """
    int: The maximal number of clusters of sentence to compute with Silhouette score.
    """

    number_of_clusters: Union[int, str] = "auto"
    """
    int: The number of wanted clusters (ie, opinion). If "auto", auto-estimation with Silhouette Score.
    """

    min_feature_per_cluster: int = 5
    """
    int: The minimal number of feature per individual cluster.
    """

@dataclass
class Parameters:

    embedding: Embedding = Embedding()

    database: Database = Database()

    preprocessing: PreProcessing = PreProcessing()

    query: Query = Query()

    first_launch: bool = True
