import re


def clean_text(text: str) -> str:
    text = re.sub("'", "", text)
    text = re.sub("(\\W)+", " ", text)
    return text