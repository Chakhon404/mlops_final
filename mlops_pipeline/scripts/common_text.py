import re

URL = re.compile(r"http\S+|www\.\S+")
MENTION = re.compile(r"@\w+")
HASHTAG = re.compile(r"#\w+")
MULTISPACE = re.compile(r"\s+")

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = URL.sub(" ", s)
    s = MENTION.sub(" ", s)
    s = HASHTAG.sub(" ", s)
    s = re.sub(r"[^a-z0-9\s\.\,\!\?\-\'\"]", " ", s)
    s = MULTISPACE.sub(" ", s).strip()
    return s
