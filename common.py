import re

def clean(t: str) -> str:
    t = str(t)
    t = re.sub(r"^.{0,80}?\(Reuters\)\s*-?\s*", "", t)  # remove dataset leakage
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"[^a-zA-Z ]", " ", t).lower()
    return re.sub(r"\s+", " ", t).strip()