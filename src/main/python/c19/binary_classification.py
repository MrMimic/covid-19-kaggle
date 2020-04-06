#!/usr/bin/env python3

import os
import time
from typing import List
import numpy as np

import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from c19.embedding import Embedding
from c19.text_preprocessing import preprocess_text


class BinaryClassifier():
    def __init__(self, model_path: str, stat_sentences: List[str],
                 other_sentences: List[str],
                 embedding_model: Embedding,
                 embedding_dimension: int) -> None:
        """
        Define a binary classifier (MLP).
        It will be able to segregate statistical conclusions sentences from others.

        Args:
            model_path (str): Path of the model to load or save.
            stat_sentences (List[str]): Positives sentences (statistical conclusions).
            other_sentences (List[str]): Negative sentences (not in interest).
            embedding_model (Embedding): The embedding model to get sentences as vectors.
            embedding_dimension (int): Embedding size for the number of neuron in HL.
        """
        self.embedding_model = embedding_model
        self.model_path = model_path

        if os.path.isfile(model_path):
            self.model_found = True
            self.load()

        else:
            self.model_found = False

            self.stat_sentences = stat_sentences
            self.other_sentences = other_sentences
            self.prepare_data()

            self.embedding_dimension = embedding_dimension
            self.prepare_model()

    def prepare_data(self) -> None:
        """
        Prepare X and y data.
        Preprocess sentences and compute vector.
        """
        # Index 0 cause only one sentences per line
        self.X = [
            self.embedding_model.compute_sentence_vector(
                preprocess_text(sentence)[0][0])
            for sentence in self.stat_sentences + self.other_sentences
        ]
        self.y = [1 for sentence in self.stat_sentences
                  ] + [0 for sentence in self.other_sentences]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.33, random_state=42)

    def compute_accuracy(self, confusion_matrix: np.array):
        """
        Total number of TP divided by the total length of X_test.
        """
        return confusion_matrix.trace() / confusion_matrix.sum()

    def prepare_model(self) -> None:
        """
        Instanciate an SK learn MLP.
        """
        self.model = MLPClassifier(
            hidden_layer_sizes=(self.embedding_dimension,
                                self.embedding_dimension // 2,
                                self.embedding_dimension // 4, 2),
            max_iter=300,
            activation="relu",
            solver="adam",
            random_state=42)

    def train_or_load(self) -> None:
        """
        Train the instanciated model.
        """
        if self.model_found is False:
            tic = time.time()
            self.model.fit(self.X_train, self.y_train)
            self.test()
            self.save()
            toc = time.time()
            self.training_time = round((toc - tic) / 60, 2)
            self.print_training_report(model_source="trained")
        else:
            self.print_training_report(model_source="loaded")

    def save(self) -> None:
        """
        Save the trained model.
        """
        joblib.dump(self.model, self.model_path)

    def load(self) -> None:
        """
        Load a model if the file_path provided to the instance exists.
        """
        self.model = joblib.load(self.model_path)

    def test(self) -> None:
        """
        Test the model and compute confusion matrix.
        """
        tic = time.time()
        y_pred = self.model.predict(self.X_test)
        self.confusion_matrix = confusion_matrix(y_pred, self.y_test)
        self.accuracy = self.compute_accuracy(self.confusion_matrix)
        self.tn, self.fp, self.fn, self.tp = self.confusion_matrix.ravel()
        toc = time.time()
        self.testing_time = round((toc - tic) / 60, 2)

    def print_training_report(self, model_source: str = "trained") -> None:
        """
        Print a report about the training.
        """
        activation = self.model.get_params()["activation"]
        solver = self.model.get_params()["solver"]
        hl_size = ", ".join([
            str(size) for size in self.model.get_params()["hidden_layer_sizes"]
        ])
        model_df = pd.DataFrame(
            {
                "Activation": [activation],
                "Solver": [solver],
                "Layers": [hl_size]
            },
            index=[""]).T.head()

        print(f"Model {model_source}: {self.model_path}.")
        print(f"{model_df}\n")

        if model_source == "trained":
            str_acc = round(self.accuracy * 100, 3)

            cm_df = pd.DataFrame(
                {
                    "TP": [self.tp],
                    "TN": [self.tn],
                    "FP": [self.fp],
                    "FN": [self.fn]
                },
                index=[""]).head()
            print(f"Training time: {self.training_time} minutes.")
            print(f"Testing time: {self.testing_time} minutes.\n")
            print(f"Accuracy: {str_acc} %.")
            print(f"Confusion matrix:\n")
            print(f"{cm_df}\n")
