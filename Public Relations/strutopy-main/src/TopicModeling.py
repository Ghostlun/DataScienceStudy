import pandas as pd
import os

input_path = 'dataFile.csv'

df = pd.read_csv('dataFile.csv')

# UnSupoerVised learning

import re
# text removeing
df['paper_text_processed'] = df['Abstract'].map(lambda x: re.sub(r'[,\.!?]', '', x) if isinstance(x, str) else x)

# Convert the text to lowercase
df['paper_text_processed'] = df['paper_text_processed'].map(lambda x: x.lower() if isinstance(x, str) else x)
df['paper_text_processed'] = df['paper_text_processed'].astype(str)
print(df['paper_text_processed'].head())

# Exploratory Analysis
from wordcloud import WordCloud

long_string = ','.join(list(df['paper_text_processed'].values))

wordCloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
wordCloud.generate(long_string)
image_path = 'wordcloud_image.png'
wordCloud.to_file(image_path)

print(f"Word cloud image saved to {image_path}")

# Prepare data for LDA Analysis

import gensim
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

def sent_to_wrods(sentences):
    for setence in sentences:
        yield(gensim.utils.simple_preprocess(str(setence), deacc=True))
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]

data = df['paper_text_processed'].values.tolist()
data_words = list(sent_to_wrods(data))
# remove stop words
data_words = remove_stopwords(data_words)
print(data_words[:1][0][:30])


import gensim.corpora as corpora
# Create Dictionary
id2word = corpora.Dictionary(data_words)
# Create Corpus
texts = data_words
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# View
print(corpus[:1][0][:30])


from pprint import pprint
# number of topics
num_topics = 10
# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics)
# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]