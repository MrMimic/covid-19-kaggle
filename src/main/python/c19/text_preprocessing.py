#!/usr/bin/env python3

import json
import multiprocessing as mp
import os
import re
import sqlite3
import time
from typing import Any, List, Tuple

import tqdm
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from retry import retry

from c19.database_utilities import get_all_articles_data, insert_rows


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

    def do_stemming(sentence: List[str],
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
        sentences = [do_stemming(sentence) for sentence in sentences]
    if remove_num is True:
        sentences = [remove_numeric_words(sentence) for sentence in sentences]
    # Filter empty sentences and one-letters words
    sentences = [[word for word in sentence if len(word) > 1]
                 for sentence in sentences if sentence != []]
    return sentences, sentences_raw


@retry(sqlite3.OperationalError, tries=5, delay=2)
def pre_process_articles(args: List[Any]) -> None:
    """
    Apply preprocessing to articles and store result into the SQLite DB.
    Retry if the select fail due to a locked database.

    Args:
        args (List[Any]): The article data to be pre-processed. See below variable attribution.
    """
    article_id: str = args[0][0]
    article_title: str = args[0][1]
    article_abstract: str = args[0][2]
    article_body: str = args[0][3]

    embedding_model = args[1]
    stem_words: bool = args[2]
    remove_num: bool = args[3]

    article_rows = []

    for section, data in zip(["title", "abstract", "body"],
                             [article_title, article_abstract, article_body]):
        if data is not None:
            pp_sentences, sentences_raw = preprocess_text(
                data, stem_words=stem_words, remove_num=remove_num)
            for pp_sentence, raw_sentence in zip(pp_sentences, sentences_raw):
                if pp_sentence != []:
                    if embedding_model is not None:
                        vector = json.dumps((*map(
                            str,
                            embedding_model.compute_sentence_vector(
                                pp_sentence)), ))
                    else:
                        vector = None
                    try:
                        row_to_insert = [
                            article_id, section, raw_sentence,
                            json.dumps(pp_sentence), vector
                        ]
                        article_rows.append(row_to_insert)
                    except TypeError:  # When all words are not in the model
                        continue
    return article_rows


def pre_process_and_vectorize_texts(embedding_model: Any,
                                    db_path: str = "articles_database.sqlite",
                                    first_launch: bool = False,
                                    stem_words: bool = False,
                                    remove_num: bool = False) -> None:
    """
    Main function allowing to pre-process every article which have been stored in the DB.

    Args:
        embedding_model (Embedding): The embedding model to be used to vectorize sentences.
        db_path (str, optional): Path to the newly created DB. Defaults to "articles_database.sqlite".
        first_launch (bool, optional): Debug option preventing to create a new file. Defaults to False.
        stem_words (bool, optional): Stem words during preprocessing. Defaults to False.
        remove_num (bool, optional): Remove numerical values during preprocessing. Defaults to False.
    """
    if first_launch is False:
        assert os.path.isfile(db_path)
        print(f"DB {db_path} will be used instead.")

    else:

        # Get all previously inserted IDS as well as a pointer on embedding method
        ids = [(article, embedding_model, stem_words, remove_num)
               for article in get_all_articles_data(db_path=db_path)]
        print(f"{len(ids)} files to pre-process.")

        #Pre-process articles
        tic = time.time()
        pool = mp.Pool(processes=8)
        rows_to_insert = list(
            tqdm.tqdm(pool.imap_unordered(pre_process_articles, ids),
                      total=len(ids),
                      desc="PRE-PROCESSING: "))
        toc = time.time()
        print(
            f"Took {round((toc-tic) / 60, 2)} min to pre-process {len(ids)} articles."
        )
        time.sleep(0.5)

        # And insert clean data
        tic = time.time()
        inserted_sentences = 0
        with tqdm.tqdm(total=len(rows_to_insert), desc="INSERTION: ") as pbar:
            for article_sentences in rows_to_insert:
                insert_rows(list_to_insert=article_sentences,
                            table_name="sentences",
                            db_path=db_path)
                inserted_sentences += len(article_sentences)
                pbar.update()
        toc = time.time()
        print(
            f"Took {round((toc-tic) / 60, 2)} min to insert {inserted_sentences} sentences (SQLite DB: {db_path})."
        )
