#!/usr/bin/env python3

import pandas as pd
from language_detector import detect_language


def update_languages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Update the language columns of the DF with the language of article.

    Args:
        df (pd.DataFrame): The dataframe, with a columns "title".

    Returns:
        pd.DataFrame: DF with a new column "lang".
    """
    languages = []
    for index, row in df.iterrows():
        try:
            lang = detect_language(row.title)[0:2]
        except Exception:
            lang = None
        languages.append(lang)
    df['lang'] = languages
    del languages

    return df
