#!/usr/bin/env python3

import json
import multiprocessing as mp
import os
import re
import sqlite3
import time
from random import shuffle
from typing import Any, List, Tuple

import tqdm
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize
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
    # Remove shortest sentences
    sentences = [sentence for sentence in sentences_raw if len(sentence) > 20]
    # Lower them
    sentences = [sentence.lower() for sentence in sentences]
    # Split sentences into words and remove punctuation
    sentences = [word_tokenize(sentence) for sentence in sentences]
    # Remove stopwords
    sentences = [filter_stopwords(sentence) for sentence in sentences]
    if stem_words is True:
        # Stem words
        sentences = [do_stemming(sentence) for sentence in sentences]
    if remove_num is True:
        sentences = [remove_numeric_words(sentence) for sentence in sentences]
    # Filter empty sentences and one or two-letters words
    sentences = [[word for word in sentence if len(word) > 2]
                 for sentence in sentences if sentence != []]
    # Remove sentence with less than 4 words
    sentences = [sentence for sentence in sentences if len(sentence) > 3]

    return sentences, sentences_raw


@retry(sqlite3.OperationalError, tries=5, delay=2)
def pre_process_batch_of_articles(args: List[Any]) -> None:
    """
    Apply preprocessing to articles and store result into the SQLite DB.
    Retry if the select fail due to a locked database.

    Args:
        args (List[Any]): The article data to be pre-processed.
        (batch, index, embedding_model, stem_words, remove_num)
    """

    embedding_model = args[1]
    stem_words: bool = args[2]
    remove_num: bool = args[3]
    max_body_sentences: int = args[4]

    articles_rows = []

    # All abstract starts with this word
    words_to_filter = ["abstract"]
    # This is several variations of the same word
    covid_regex = re.compile(r"([Cc][Oo][Vv][Ii][Dd][- ]?[12][90](19)?)")
    coronavirus_regex = re.compile(
        r"[Cc][Oo][Rr][Oo][Nn][Aa] [Vv][Ii][Rr][Uu][Ss]")

    for article in args[0]:

        article_id: str = article[0]
        article_title: str = article[1]
        article_abstract: str = article[2]
        article_body: str = article[3]

        for section, data in zip(
            ["title", "abstract", "body"],
            [article_title, article_abstract, article_body]):
            if data is not None:
                # Replace synonyms
                data = re.sub(covid_regex, "COVID-19", data)
                data = re.sub(coronavirus_regex, "Coronavirus", data)
                pp_sentences, sentences_raw = preprocess_text(
                    data, stem_words=stem_words, remove_num=remove_num)
                if len(pp_sentences) > 0:
                    # HDD issue: let's randomly select max_body_sentences sentences.
                    if section == "body" and max_body_sentences > 0 and len(
                            pp_sentences) > max_body_sentences:
                        temp_list = list(zip(pp_sentences, sentences_raw))
                        shuffle(temp_list)
                        pp_sentences, sentences_raw = zip(
                            *temp_list[0:max_body_sentences])
                        del temp_list
                    for pp_sentence, raw_sentence in zip(
                            pp_sentences, sentences_raw):
                        # Only keep is sentence has at least 4 words.
                        if len(pp_sentence) > 4:
                            # Filter some words
                            if pp_sentence[0] in words_to_filter:
                                pp_sentence.pop(0)
                            if embedding_model is not None:
                                try:
                                    vector = json.dumps((*map(
                                        str,
                                        embedding_model.
                                        compute_sentence_vector(pp_sentence)),
                                                         ))
                                except TypeError:
                                    vector = None
                            else:
                                vector = None
                            row_to_insert = [
                                article_id, section, raw_sentence,
                                json.dumps(pp_sentence), vector
                            ]
                            articles_rows.append(row_to_insert)
    return articles_rows


def split_into_chunks(iterable, chunks_size=1):
    """
    Split an iterable into chunks of size chunks.

    Args:
        iterable ([type]): List to split.
        chunks_size (int, optional): Size of each chunk. Defaults to 1.

    Returns:
        [type]: [description]
    """
    batches = []
    total_size = len(iterable)
    for ndx in range(0, total_size, chunks_size):
        batches.append(iterable[ndx:min(ndx + chunks_size, total_size)])
    return batches


def pre_process_and_vectorize_texts(embedding_model: Any,
                                    db_path: str = "articles_database.sqlite",
                                    first_launch: bool = False,
                                    stem_words: bool = False,
                                    remove_num: bool = False,
                                    batch_size: int = 1000,
                                    max_body_sentences: int = 10) -> None:
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
        articles = [
            article for article in get_all_articles_data(db_path=db_path)
        ]
        batches = split_into_chunks(articles, chunks_size=batch_size)
        arguments = [(batch, embedding_model, stem_words, remove_num,
                      max_body_sentences) for batch in batches]
        print(
            f"{len(articles)} files to pre-process ({len(batches)} batches of {len(batches[0])} articles)."
        )
        del articles
        del batches

        # Pre-process articles
        tic = time.time()
        pool = mp.Pool(processes=os.cpu_count())
        batches_to_insert = list(
            tqdm.tqdm(pool.imap_unordered(pre_process_batch_of_articles,
                                          arguments),
                      total=len(arguments),
                      desc="PRE-PROCESSING: "))
        toc = time.time()
        print(
            f"Took {round((toc-tic) / 60, 2)} min to pre-process {len(batches_to_insert)} batches of articles."
        )
        del arguments
        time.sleep(0.5)

        # And insert clean as batch as well data
        tic = time.time()
        inserted_sentences = 0
        for article_sentences in batches_to_insert:
            insert_rows(list_to_insert=article_sentences,
                        table_name="sentences",
                        db_path=db_path)
            inserted_sentences += len(article_sentences)
        toc = time.time()
        del batches_to_insert
        time.sleep(0.5)

        print(
            f"Took {round((toc-tic) / 60, 2)} min to insert {inserted_sentences} sentences (SQLite DB: {db_path})."
        )
