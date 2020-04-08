#!/usr/bin/env python3

import json
import os
import sqlite3
import time
from typing import List, Dict
import pandas as pd

import joblib
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class TfIdf():
    """
    Simple TFIDF class.
    """
    def __init__(self, max_df: int = 0.95, min_df: int = 1):
        self.max_df = max_df
        self.min_df = min_df

    def train(self, sentences) -> None:
        """
        Train the TF-IDf model.
        """
        tic = time.time()
        # Not word appearing in > 95% of docs, and less than 10 times in total
        self.counter = CountVectorizer(max_df=self.max_df, min_df=self.min_df)
        self.count_vector = self.counter.fit_transform(sentences)

        self.tfidf = TfidfTransformer(smooth_idf=True, use_idf=True)
        self.tfidf.fit_transform(self.count_vector)
        toc = time.time()
        print(f"Training TF-IDF took: {round((toc-tic) / 60, 3)} minutes.")

    def get_score(self, word: str) -> float:
        """
        Return the TFIDF score of a word.

        Args:
            word (str): The word.

        Returns:
            float: The score.
        """
        try:
            index = self.counter.get_feature_names().index(word)
            score = self.tfidf.idf_[index]
        except (ValueError, IndexError):  # If word was not in the TF-IDF
            score = 1
        return score


class W2V():
    """
    Allows to rapidly train a word2vec model on the Kaggle dataset.
    Pre-trained vectors were nice, but way too general.
    """
    def __init__(self, db_path: str, tfidf_path: str, w2v_path: str, w2v_params: Dict, parquet_output_path: str):
        self.db_path = db_path
        self.tfidf_path = tfidf_path
        self.w2v_path = w2v_path
        self.w2v_params = w2v_params
        self.parquet_output_path = parquet_output_path

    def get_sentences(self) -> List[str]:
        """
        Return all pre-processed sentences from a previous version of the base.
        Lists of tokens are re-joined by a space.
        """
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()
        cursor.execute("SELECT sentence FROM sentences")
        sentences = [
            " ".join(json.loads(sentence[0]))
            for sentence in cursor.fetchall()
        ]
        cursor.close()
        connection.close()
        print(f"Training Word2Vec and TF-IDF on {len(sentences)} sentences.")

        return sentences

    def train(self) -> None:
        """
        Train the word2vec model on these sentences.
        """
        # Get sentences as List of str
        sentences = self.get_sentences()

        # Train TFIDF
        tfidf = TfIdf()
        tfidf.train(sentences)
        joblib.dump(tfidf, self.tfidf_path)

        # Split sentences into List of List of words
        sentences = [sentence.split() for sentence in sentences]

        # Train Word2Vec
        tic = time.time()
        if self.w2v_params is None:
            print("W2v will be trained with default arguments.")
            self.model = Word2Vec(sentences)
        else:
            self.model = Word2Vec(sentences, **self.w2v_params)
        del sentences
        self.model.wv.save_word2vec_format(self.w2v_path, binary=True)
        self.merge_output(tfidf)
        toc = time.time()
        print(f"Training Word2Vec took: {round((toc-tic) / 60, 3)} minutes.")

    def merge_output(self, tfidf):
        """
        Merge TF-IDF scores and vectors into a parquet file.
        """
        words = self.model.wv.vocab.keys()
        vectors = [self.model[word] for word in words]
        scores = [tfidf.get_score(word) for word in words]
        merged_df = pd.DataFrame({
            "tfidf": scores,
            "vector": vectors
        },
                                 index=words)
        merged_df.to_parquet(self.parquet_output_path, engine="pyarrow", index=True)

    def load(self, file_path: str) -> None:
        """
        Load a word2vec binary model.

        Args:
            file_path (str): Path to the .bin file.
        """
        self.model = KeyedVectors.load_word2vec_format(file_path, binary=True)
        print(f"Loaded model containing {len(self.model.wv.vocab)} words.")
