#!/usr/bin/env python3

import json
import os
import sqlite3
import time
from typing import Any, Dict, List

import numpy as np

from gensim.models import KeyedVectors, Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import stopwords
from text_preprocessing import preprocess_text


class Embedding():
    def __init__(self,
                 model_type: str,
                 vectors_path: str,
                 embeddings_dimension: int = 50,
                 sentence_embedding_method: str = "mowe"):
        """
        The goal here is to create a self.vectors: {word: vector} regardless the model type (glove, word2vec, etc).
        """
        self.vectors_path = vectors_path
        self.embeddings_dimension = embeddings_dimension
        self.sentence_embedding_method = sentence_embedding_method
        self.model_type = model_type

        self.stop_words = set(stopwords.words("english"))

        self.vectors = {}

        if self.model_type == "glove":
            self.build_glove_vectors()
        elif "word2vec" in self.model_type:
            self.build_word2vec_vectors()
        else:
            raise Exception(f"Unknown embedding model type: {model_type}")

    def build_word2vec_vectors(self) -> None:
        """
        Load word2vec vectors into the self.vectors object.
        """
        tic = time.time()

        model = KeyedVectors.load_word2vec_format(self.vectors_path,
                                                  binary=True)

        for token in model.vocab:
            # Prevent to keep useless words (otherwise pre-proc return nothing)
            word, word_raw = preprocess_text(token)
            del word_raw
            if word:
                try:
                    if word[0][0] not in self.stop_words and len(
                            word[0][0]) > 2:
                        self.vectors[word[0][0]] = list(model[token])
                except (
                        KeyError, IndexError
                ):  # The word is not in the model or not return by the pre-proc
                    continue
        del model
        toc = time.time()
        print(
            f"Took {round((toc-tic) / 60, 2)} min to load {len(self.vectors.keys())} Word2Vec vectors (embedding dim: {self.embeddings_dimension})."
        )

    def build_glove_vectors(self) -> None:
        """
        Load GloVe vectors into the self.vectors object.
        """
        tic = time.time()

        with open(self.vectors_path, "r") as handler:
            for line in handler.readlines():
                try:
                    # Prevent to keep useless words (otherwise pre-proc return nothing)
                    word, word_raw = preprocess_text(line.split()[0])
                    vector = [
                        float(dimension) for dimension in line.split()[1:None]
                    ]
                    assert len(vector) == self.embeddings_dimension
                    self.vectors[word[0][0]] = vector
                except IndexError:  # When the preprocessing does not return a word (useless word)
                    continue

        toc = time.time()
        print(
            f"Took {round((toc-tic) / 60, 2)} min to load {len(self.vectors.keys())} GloVe vectors (embedding dim: {self.embeddings_dimension})."
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
        words_vector = [
            self.vectors[word] if word in self.vectors.keys() else list(
                list(np.full([
                    1,
                    self.embeddings_dimension,
                ], np.nan))[0]) for word in sentence
        ]
        if self.sentence_embedding_method == "mowe":
            sentence_embedding = np.nanmean(words_vector, axis=0)
        elif self.sentence_embedding_method == "sowe":
            sentence_embedding = np.nansum(words_vector, axis=0)
        else:
            raise Exception(
                f"No such sentence embedding method: {sentence_embedding_method}"
            )
        return sentence_embedding


class W2V():
    """
    Allows to rapidly train a word2vec model on the Kaggle dataset.
    Pre-trained vectors were nice, but way too general.
    """
    def __init__(
        self,
        db_path: str = os.path.join(
            os.sep,
            "content",
            "drive",
            "My Drive",
            "COVID_19_KAGGLE_DB",
            "articles_database_v3_03042020_embedding_100_remove_num_True_stem_words_False.sqlite",
        )):
        self.db_path = db_path

    def get_sentences(self) -> None:
        """
        Return all pre-processed sentences from a previous version of the base.
        """
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()
        cursor.execute("SELECT sentence FROM sentences")
        self.sentences = [
            json.loads(sentence[0]) for sentence in cursor.fetchall()
        ]
        cursor.close()
        connection.close()
        print(f"Training Word2Vec on {len(self.sentences)} sentences.")

    def train(self, file_path: str = None) -> None:
        """
        Train the word2vec model on these sentences.
        Param have been set up with: https://www.aclweb.org/anthology/W16-2922.pdf

        Args:
            file_path (str, optional): File path to save vectors. Defaults to None.
        """
        self.get_sentences()

        tic = time.time()
        self.model = Word2Vec(self.sentences,
                              sg=1,
                              hs=1,
                              sample=1e-5,
                              negative=10,
                              min_count=3,
                              size=200,
                              window=7,
                              seed=42,
                              workers=os.cpu_count(),
                              iter=10)
        toc = time.time()

        print(f"Training took: {round((toc-tic) / 60, 3)} minutes.")

        if file_path is None:
            model_path = f"word2vec_on_kaggle_corpus_dim_{self.model.vector_size}_min_count_{self.model.min_count}_vocab_{len(self.model.wv.vocab)}.bin"
        self.model.wv.save_word2vec_format(model_path, binary=True)

    def load(self, file_path: str) -> None:
        """
        Load a word2vec binary model.

        Args:
            file_path (str): Path to the .bin file.
        """
        self.model = KeyedVectors.load_word2vec_format(file_path, binary=True)

        print(f"Loaded model containing {len(self.model.wv.vocab)} words.")
