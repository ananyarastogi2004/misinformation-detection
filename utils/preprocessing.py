import re


def clean_text(text: str) -> str:
    """
    Basic text preprocessing
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)  # remove extra spaces
    text = text.strip()
    return text