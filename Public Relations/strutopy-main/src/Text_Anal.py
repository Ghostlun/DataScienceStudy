import os
import re
import pandas as pd
from collections import Counter

# Define paths
artifacts_dir = "artifacts/wiki_data"
input_path = "PublicRelation.csv"
stop_words_path = os.path.join("artifacts", "auxiliary_data", "stop_words_english.txt")

# Load data
df = pd.read_csv(input_path, index_col=0)

# Preprocess the text in the "Abstract" column
text_corpus = list(df["Abstract"].astype(str))
# Remove punctuation
text_corpus = [re.sub(r"[^\w\s]", "", doc) for doc in text_corpus]
# Remove digits
text_corpus = ["".join([i for i in doc if not i.isdigit()]) for doc in text_corpus]

# Load stopwords
with open(stop_words_path) as f:
    stoplist = f.read().split()

# Remove stopwords and clean the text
texts = [
    [word for word in document.lower().split() if word not in stoplist]
    for document in text_corpus
]

# Extract the "Author Keywords" column and drop any NaN values
keywords_series = df["Author Keywords"].dropna()

# Split the keywords by the delimiter (assuming it's a comma)
keywords_list = keywords_series.str.split(',').tolist()

# Flatten the list of lists into a single list and remove stopwords
flattened_keywords = [keyword.strip().lower() for sublist in keywords_list for keyword in sublist if keyword.strip().lower() not in stoplist]

# Combine the cleaned texts and keywords
combined_words = [word for text in texts for word in text] + flattened_keywords

# Count the frequency of each word
word_counts = Counter(combined_words)

# Convert the Counter object to a DataFrame for easier viewing
word_counts_df = pd.DataFrame(word_counts.items(), columns=['Word', 'Count']).sort_values(by='Count', ascending=False)

# Display the DataFrame
print(word_counts_df)
print(list(word_counts_df['Word']))