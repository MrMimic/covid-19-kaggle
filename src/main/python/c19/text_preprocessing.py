#!/usr/bin/env python3

import json
import os
import re
import sqlite3
import time
from typing import Any, List, Tuple

from c19.database_utilities import get_all_articles_doi, insert_row
from c19.embedding import Embedding
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from pathos.multiprocessing import ProcessingPool as picklable_pool


def preprocess_text(text: str,
                    stem_words: bool = False,
                    remove_num: bool = True) -> Tuple[List[str], List[str]]:
    """
    Pre-process a text. Remove stop words, lowerise, tokenise, etc.

    Args:
        text (str): The text to be pre-processed.
        stem_words (bool, optional): Stem words or not. Defaults to False.
        remove_num (bool, optional): Remove numerics values or not. Defaults to True.

    Returns:
        Tuple[List[str], List[str]]: Two lists: raw and pre-processed sentences.
    """
    word = RegexpTokenizer(r"\w+")
    stop_words_nltk = set(stopwords.words("english"))
    stemmer = SnowballStemmer("english")

    def filter_stopwords(sentence: List[str],
                         stop_words: List[str] = stop_words_nltk) -> List[str]:
        """ Remove stopwords from a given list of words. """
        return [word for word in sentence if word not in stop_words]

    def stem_words(sentence: List[str],
                   stem_function: Any = stemmer) -> List[str]:
        """ Get words root for every member of an input list. """
        return [stem_function.stem(word) for word in sentence]

    def remove_numeric_words(sentence: List[str]) -> List[str]:
        """ Remove number (items) from a list of words. """
        letter_pattern = re.compile(r"[a-z]")
        return [word for word in sentence if letter_pattern.match(word)]

    # Split paragraphs into sentences and keep them for nive output
    sentences_raw = sent_tokenize(text)
    # Lower
    sentences = [sentence.lower() for sentence in sentences_raw]
    # Split sentences into words and remove punctuation
    sentences = [word.tokenize(sentence) for sentence in sentences]
    # Remove stopwords
    sentences = [filter_stopwords(sentence) for sentence in sentences]
    if stem_words is True:
        # Stem words
        sentences = [stem_words(sentence) for sentence in sentences]
    if remove_num is True:
        sentences = [remove_numeric_words(sentence) for sentence in sentences]
    # Filter empty sentences and one-letters words
    sentences = [[word for word in sentence if len(word) > 1]
                 for sentence in sentences if sentence != []]
    return sentences, sentences_raw


def pre_process_articles(args: List[Any]) -> None:
    """
    Apply preprocessing to articles and store result into the SQLite DB.

    Args:
        args (List[Any]): The article data to be pre-processed. See below variable attribution.
    """
    article_id: str = args[0]
    embedding_model = args[1]
    db_path: str = args[2]
    stem_words: bool = args[3]
    remove_num: bool = args[4]

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM articles WHERE paper_doi = ?", [article_id])
    # Get dict {column: value}
    try:
        article = {
            [col for col in head if col is not None][0]: value
            for head, value in zip(cursor.description, cursor.fetchone())
        }
        cursor.close()
        connection.close()
    except TypeError:  # When the DB doest not return a result
        cursor.close()
        connection.close()
        return None

    for section in ["title", "abstract", "body"]:
        if article[section] is not None:
            pp_sentences, sentences_raw = preprocess_text(
                article[section], stem_words=stem_words, remove_num=remove_num)
            for pp_sentence, raw_sentence in zip(pp_sentences, sentences_raw):
                try:
                    # paper, section, sentence, vector
                    row_to_insert = [
                        article_id,
                        section,
                        raw_sentence,  # Raw sentence
                        json.dumps(pp_sentence
                                   ),  # Store list of tokens as loadable str
                        json.dumps([
                            str(x)
                            for x in embedding_model.compute_sentence_vector(
                                pp_sentence)
                        ])  # Embeded vector
                    ]
                    try:
                        insert_row(list_to_insert=row_to_insert,
                                   table_name="sentences",
                                   db_path=db_path)
                    except sqlite3.OperationalError:  # Even the retry() decorator failed
                        continue
                except TypeError:  # When all words are not in the model
                    continue


def pre_process_and_vectorize_texts(embedding_model: Embedding,
                                    db_path: str = "articles_database.sqlite",
                                    load_file: bool = True,
                                    stem_words: bool = False,
                                    remove_num: bool = False) -> None:
    """
    Main function allowing to pre-process every article which have been stored in the DB.

    Args:
        embedding_model (Embedding): The embedding model to be used to vectorize sentences.
        db_path (str, optional): Path to the newly created DB. Defaults to "articles_database.sqlite".
        load_file (bool, optional): Debug option preventing to create a new file. Defaults to True.
        stem_words (bool, optional): Stem words during preprocessing. Defaults to False.
        remove_num (bool, optional):Remove numerical values during preprocessing. Defaults to False.
    """
    if load_file is True:
        assert os.path.isfile(db_path)
        print(f"DB {db_path} will be used instead.")

    else:
        tic = time.time()

        # Get all previously inserted IDS as well as a pointer on embedding method
        ids = [(id_, embedding_model, db_path, stem_words, remove_num)
               for id_ in get_all_articles_doi(db_path=db_path)]
        with picklable_pool(os.cpu_count()) as pool:
            pool.map(pre_process_articles, ids)

        # Only to count pp sentences
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        cursor.execute("SELECT paper_doi FROM sentences")
        sentences = cursor.fetchall()
        cursor.close()
        connection.close()

        toc = time.time()
        print(
            f"Took {round((toc-tic) / 60, 2)} min to pre-process {len(ids)} articles with {len(sentences)} sentences (SQLite DB: {db_path})."
        )
