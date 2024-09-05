
import os
import re
from pydoc import doc

import numpy as np
import pandas as pd
from gensim import corpora
from gensim.parsing.preprocessing import STOPWORDS
# 이 Python 스크립트는 텍스트 데이터를 전처리하고, 전처리된 데이터를 이용해 Bag of Words (BoW) 모델을 생성하는 과정을 보여줍니다. 이 과정은 자연어 처리(NLP)에서 주로 사용됩니다. 각 부분을 설명하겠습니다:

artifacts_dir = "artifacts/wiki_data"

input_path = "PublicRelation.csv"
stop_words_path = os.path.join("artifacts", "auxiliary_data", "stop_words_english.txt")


df = pd.read_csv(input_path, index_col=0)

text_corpus = list(df["Abstract"].astype(str))
# Remove Functuation
text_corpus = [re.sub(r"[^\w\s]", "", doc) for doc in text_corpus]
# Remove isDegit
text_corpus = ["".join([i for i in doc if not i.isdigit()]) for doc in text_corpus]

# Create a set of frequent words
with open(stop_words_path) as f:
    stoplist = f.read().split()

print(stoplist)
texts = [
    [word for word in document.lower().split() if word not in stoplist]
    for document in text_corpus
]

df["text_preproc"] = texts

dictionary = corpora.Dictionary()
BoW_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in texts]

corpora.MmCorpus.serialize("artifacts/wiki_data/BoW_corpus.mm", BoW_corpus)
dictionary.save("artifacts/wiki_data/dictionary.mm")
df.to_csv("artifacts/wiki_data/corpus_preproc.csv")


