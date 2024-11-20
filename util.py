import re


def is_not_german(text):
    pattern = re.compile(r"[^a-zA-ZäöüÄÖÜß.,:;!?'\(\)\- ]")

    return bool(re.search(pattern, text))
