#!/usr/bin/env python3

import json
import os
import sqlite3
import time
from typing import List

import joblib
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class TfIdf():
    """
    Simple TFIDF class.
    """
    def __init__(self, sentences):
        # Not word appearing in > 95% of docs, and less than 10 times in total
        self.counter = CountVectorizer(max_df=0.95, min_df=10)
        self.count_vector = self.counter.fit_transform(sentences)

        self.tfidf = TfidfTransformer(smooth_idf=True, use_idf=True)
        self.tfidf.fit_transform(self.count_vector)

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

    def get_sentences(self) -> List[str]:
        """
        Return all pre-processed sentences from a previous version of the base.
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

    def train(self, file_path: str = None) -> None:
        """
        Train the word2vec model on these sentences.
        Param have been set up with: https://www.aclweb.org/anthology/W16-2922.pdf

        Args:
            file_path (str, optional): File path to save vectors. Defaults to None.
        """
        tfidf_model_path = "tfidf_on_kaggle_corpus_v3.pkl"
        if file_path is not None:
            tfidf_model_path = os.path.join(os.path.dirname(file_path),
                                            tfidf_model_path)

        # Get sentences as List of str
        sentences = self.get_sentences()

        # Train TFIDF
        tic = time.time()
        tfidf = TfIdf(sentences)
        joblib.dump(tfidf, tfidf_model_path)
        toc = time.time()
        print(f"Training TF-IDF took: {round((toc-tic) / 60, 3)} minutes.")

        # Split sentences into List of List of words
        sentences = [sentence.split() for sentence in sentences]

        # Train Word2Vec
        tic = time.time()
        self.model = Word2Vec(sentences,
                              sg=1,
                              hs=1,
                              sample=1e-5,
                              negative=10,
                              min_count=10,
                              size=100,
                              window=7,
                              seed=42,
                              workers=os.cpu_count(),
                              iter=10)
        toc = time.time()
        del sentences
        print(f"Training Word2Vec took: {round((toc-tic) / 60, 3)} minutes.")

        if file_path is None:
            w2v_model_path = f"word2vec_on_kaggle_corpus_v3.bin"
        self.model.wv.save_word2vec_format(w2v_model_path, binary=True)

    def load(self, file_path: str) -> None:
        """
        Load a word2vec binary model.

        Args:
            file_path (str): Path to the .bin file.
        """
        self.model = KeyedVectors.load_word2vec_format(file_path, binary=True)

        print(f"Loaded model containing {len(self.model.wv.vocab)} words.")
