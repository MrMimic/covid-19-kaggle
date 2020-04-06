#!/usr/bin/env python3

from textblob import TextBlob


def get_lang(text: str) -> str:
    """
    Detects language of text: must contain minimum 3 characters.

    Args:
        text (str): The text from which language should be detected.

    Raises:
        ValueError: The provided string is not long enough.

    Returns:
        str: [description]
    """
    if len(text) >= 3:
        blob = TextBlob(text)
        return blob.detect_language()
    else:
        raise ValueError(
            "A minimum of 3 characters are needed for language detection!")
