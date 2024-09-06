import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

file_path = '../datasets/kjv.csv'
df = pd.read_csv(file_path)

print(df.head())

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)
    tokens = ','.join([lemmatizer.lemmatize(word) for word in tokens if word not in stop_words])
    return tokens


df['tokens'] = df['text'].apply(preprocess_text)

print(df[['citation', 'tokens']].head())

df.to_csv('../datasets/kjv_tokens.csv', index=False)
