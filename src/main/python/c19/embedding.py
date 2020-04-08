#!/usr/bin/env python3

import os
import time
import urllib.request
from typing import Any, List

import numpy as np
from c19.file_processing import read_parquet


def get_pre_trained_vectors(file_path: str) -> None:
    """
    Download pre-trained vectors from github.

    Args:
        file_path (str): Path to local file to be written.
    """
    url = "https://github.com/MrMimic/covid-19-kaggle/raw/master/resources/global_df_w2v_tfidf.parquet"
    urllib.request.urlretrieve(url, file_path)


class Embedding():
    def __init__(self,
                 parquet_embedding_path: str,
                 embeddings_dimension: int = 50,
                 sentence_embedding_method: str = "mowe",
                 weight_vectors: bool = False):
        """
        The goal here is to create a self.vectors: {word: vector} regardless the model type (glove, word2vec, etc).
        Everything is loaded from a parquet file containing previously trained data.
        """
        self.parquet_embedding_path = parquet_embedding_path
        if os.path.isfile(self.parquet_embedding_path) is False:
            get_pre_trained_vectors(self.parquet_embedding_path)

        self.weight_vectors = weight_vectors
        self.embeddings_dimension = embeddings_dimension
        self.sentence_embedding_method = sentence_embedding_method
        self.vectors = {}

        self.load_word2vec_vectors()

    def load_word2vec_vectors(self) -> None:
        """
        Load word2vec vectors into the self.vectors object.
        Weight each vector object by the TFIDF score of the coresponding token.
        """
        tic = time.time()

        data_frame = read_parquet(self.parquet_embedding_path)
        for word, data in data_frame.iterrows():
            if self.weight_vectors is True:
                self.vectors[word] = self.get_weighted_vector(
                    vector=data.vector, coefficient=data.tfidf)
            else:
                self.vectors[word] = data.vector
        del data_frame

        toc = time.time()
        print(
            f"Took {round((toc-tic) / 60, 2)} min to load {len(self.vectors.keys())} Word2Vec vectors (embedding dim: {self.embeddings_dimension})."
        )

    def compute_sentence_vector(
            self,
            sentence: List[str],
            sentence_embedding_method: str = "mowe") -> List[float]:
        """
        Compute a SOWE/MOWE over all tokens composing a sentence. Word skipped if not in model.

        Args:
            sentence (List[str]): The list of word to be embeded.
            sentence_embedding_method (str, optional): Either MOWE or SOWE (mean or sum over columns). Defaults to "mowe".

        Raises:
            Exception: The sentence embedding method is different than MOWE or SOWE.

        Returns:
            List[float]: The sentence vector.
        """
        words_vector = [self.vectors[word] for word in sentence if word in self.vectors.keys()]
        if len(words_vector) > 0:
            if self.sentence_embedding_method == "mowe":
                sentence_embedding = np.mean(words_vector, axis=0)
            elif self.sentence_embedding_method == "sowe":
                sentence_embedding = np.sum(words_vector, axis=0)
            else:
                raise Exception(
                    f"No such sentence embedding method: {sentence_embedding_method}"
                )
        else:
            sentence_embedding = None
        return sentence_embedding

    def get_weighted_vector(self, vector: List[float],
                            coefficient: float) -> List[float]:
        """
        Apply a coefficient on all items of a vector.

        Args:
            vector (List[float]): Vector to be weighted.
            coefficient (float): The weight.

        Returns:
            List[float]: [description]
        """
        return list(map(lambda x: x * coefficient, vector))
