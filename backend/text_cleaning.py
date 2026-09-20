"""Shared text cleaning for training AND serving.

preprocessing.py and app.py both import clean_text from here so the two can
never drift apart - they previously had separate copies that diverged.

The regex matters: `\w` matches letters and digits but NOT combining marks,
and Devanagari/Bengali vowel signs are combining marks (category Mn/Mc). A
plain [^\w\s] therefore strips every matra, silently turning Indic text into
vowel-less nonsense. The Unicode ranges below keep those scripts intact.
"""

import re

_URL = re.compile(r"http\S+|www\S+")
_USER = re.compile(r"@\w+")
_PUNCT = re.compile(r"[^\w\sऀ-ॿঀ-৿]", re.UNICODE)  # Devanagari + Bengali
_WS = re.compile(r"\s+")


def clean_text(text):
    """Strip URLs, @mentions and punctuation. Indic scripts survive intact."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = "" if text != text else str(text)   # NaN -> ""
    text = _URL.sub("", text)
    text = _USER.sub("", text)
    text = _PUNCT.sub("", text)
    return _WS.sub(" ", text).strip()


if __name__ == "__main__":
    assert clean_text("तू बेवकूफ है") == "तू बेवकूफ है"
    assert clean_text("तू कोण लवड्या") == "तू कोण लवड्या"
    assert clean_text("তুই একটা গাধা") == "তুই একটা গাধা"
    assert clean_text("you are an idiot!") == "you are an idiot"
    assert clean_text("@bob http://x.com hi") == "hi"
    assert clean_text(None) == ""
    assert clean_text("a   b") == "a b"
    print("text_cleaning: all checks passed")
