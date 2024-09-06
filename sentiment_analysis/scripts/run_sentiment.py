import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline


file_path = '../datasets/kjv.csv'
df = pd.read_csv(file_path)

model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

sa = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)


def get_sentiment_score(text):
    result = sa(text)[0]
    return result['label'], result['score']


df['sentiment_label'], df['sentiment_score'] = zip(*df['text'].apply(get_sentiment_score))
book_sentiments = df.groupby('book')['sentiment_score'].mean().reset_index()
bible_books_order = [
    'Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy', 'Joshua', 'Judges', 'Ruth',
    '1 Samuel', '2 Samuel', '1 Kings', '2 Kings', '1 Chronicles', '2 Chronicles', 'Ezra',
    'Nehemiah', 'Esther', 'Job', 'Psalms', 'Proverbs', 'Ecclesiastes', 'Song of Solomon',
    'Isaiah', 'Jeremiah', 'Lamentations', 'Ezekiel', 'Daniel', 'Hosea', 'Joel', 'Amos',
    'Obadiah', 'Jonah', 'Micah', 'Nahum', 'Habakkuk', 'Zephaniah', 'Haggai', 'Zechariah',
    'Malachi', 'Matthew', 'Mark', 'Luke', 'John', 'Acts', 'Romans', '1 Corinthians',
    '2 Corinthians', 'Galatians', 'Ephesians', 'Philippians', 'Colossians', '1 Thessalonians',
    '2 Thessalonians', '1 Timothy', '2 Timothy', 'Titus', 'Philemon', 'Hebrews', 'James',
    '1 Peter', '2 Peter', '1 John', '2 John', '3 John', 'Jude', 'Revelation'
]
book_sentiments['book'] = pd.Categorical(book_sentiments['book'], categories=bible_books_order, ordered=True)
book_sentiments = book_sentiments.sort_values('book')
df.to_csv('../datasets/kjv_sentiment.csv', index=False)
print('Saved kjv_sentiment.csv')
book_sentiments.to_csv('../datasets/kjv_sentiment_books.csv', index=False)
print('Saved kjv_sentiment_books.csv')