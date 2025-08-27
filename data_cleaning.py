# Input file format:
# CSV (or DataFrame) with at least:

# conv_id – conversation/thread ID

# speaker – user/agent (optional but useful)

# text – the utterance

import re
import pandas as pd

df = pd.read_csv("conversations.csv")

def clean(t: str) -> str:
    t = str(t).strip()
    t = re.sub(r'https?://\S+|www\.\S+', ' ', t)   # URLs → space
    t = re.sub(r'\S+@\S+', ' ', t)                 # emails → space
    t = re.sub(r'\s+', ' ', t)
    return t

df["text_clean"] = df["text"].apply(clean)
df = df[df["text_clean"].str.len() > 0].reset_index(drop=True)
