#!/usr/bin/env python3
"""
Parameters for the Kaggle Covid19 project.
Default value can be erased as follow:

    >>> param = parameters.Parameters(query=parameters.Query(min_cluster=3))
    >>> param.query
    >>> Query(top_k_sentences=50, min_cluster=3, max_cluster=10)
"""

from dataclasses import dataclass
import os


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


@dataclass
class PreProcessing:

    load_text_body: bool = True
    """
    bool: Load the body text of article in addition to the abstract and title.
    """

    stem_words: bool = True
    """
    bool: Should stem words during pre-processing.
    """

    remove_numeric: bool = True
    """
    bool: Should remove numeric tokens during pre-processing.
    """


@dataclass
class Query:

    top_k_sentences: int = 50
    """
    int: The number of close sentences to be tajen in account when computing clustering.
    """

    min_cluster: int = 1
    """
    int: The minimal number of clusters os sentence to design.
    """

    max_cluster: int = 10
    """
    int: The maximal number of clusters os sentence to design.
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
        # Embedding
        assert os.path.exists(
            os.path.dirname(os.path.realpath(self.embedding.local_path))
        ), f"The path for the pre-trained vectors does not exists: {self.embedding.local_path}"
        assert isinstance(
            self.embedding.dimension, int
        ), f"An interger is requiered for embedding dimension: {self.embedding.dimension}"
        assert self.embedding.word_aggregation_method in [
            "mowe", "sowe"
        ], f"Word aggregation method should be either 'mowe' or 'sowe'"
        assert isinstance(
            self.embedding.weight_with_tfidf, bool
        ), f"An boolean is requiered to decide if word should be weighted or not: {self.embedding.weight_with_tfidf}"
        # Database
        assert os.path.exists(
            os.path.dirname(os.path.realpath(self.database.local_path))
        ), f"The path for the article database does not exists: {self.database.local_path}"
        assert os.path.exists(
            self.database.local_path
        ), f"The path for the article database does not exists: {self.database.local_path}"
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
            self.query.top_k_sentences, int
        ), f"An interger is requiered to determine top-k sentences: {self.query.top_k_sentences}"
        assert isinstance(
            self.query.min_cluster, int
        ), f"An interger is requiered to determine minimal number of sentences clusters: {self.query.min_cluster}"
        assert isinstance(
            self.query.max_cluster, int
        ), f"An interger is requiered to determine maximal number of sentences clusters: {self.query.max_cluster}"
