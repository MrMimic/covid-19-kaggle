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

    def __post_init__(self):
        """
        Parameters validation.
        """
        assert isinstance(
            self.first_launch, bool
        ), f"An boolean is requiered to decide if new things have to be trained or not: {self.first_launch}"
        assert isinstance(
            self.embedding.dimension, int
        ), f"An interger is requiered for embedding dimension: {self.embedding.dimension}"
        assert self.embedding.word_aggregation_method in [
            "mowe", "sowe"
        ], f"Word aggregation method should be either 'mowe' or 'sowe'"
        assert isinstance(
            self.embedding.weight_with_tfidf, bool
        ), f"An boolean is requiered to decide if word should be weighted or not: {self.embedding.weight_with_tfidf}"
        # Preprocessing
        assert isinstance(
            self.preprocessing.load_text_body, bool
        ), f"An boolean is requiered to decide if text body should be loaded or not: {self.preprocessing.load_text_body}"
        assert isinstance(
            self.preprocessing.stem_words, bool
        ), f"An boolean is requiered to decide if text should be stemmed or not: {self.preprocessing.stem_words}"
        assert isinstance(
            self.preprocessing.remove_numeric, bool
        ), f"An boolean is requiered to decide if numeric values should be removed or not: {self.preprocessing.remove_numeric}"
        # Query
        assert isinstance(
            self.query.k_min, int
        ), f"An interger is requiered to determine minimal number of sentences clusters: {self.query.k_min}"
        assert isinstance(
            self.query.k_max, int
        ), f"An interger is requiered to determine maximal number of sentences clusters: {self.query.k_max}"
