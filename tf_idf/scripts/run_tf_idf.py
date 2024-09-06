import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


file_path = '../datasets/kjv_tokens.csv'
df = pd.read_csv(file_path)

tfidf_results_chapters = []

for book in df['book'].unique():
    book_df = df[df['book'] == book]
    book_df_chapters = book_df.groupby('chapter')['tokens'].apply(lambda x: ' '.join(x)).reset_index()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(book_df_chapters['tokens'])
    feature_names = vectorizer.get_feature_names_out()
    for i, row in book_df_chapters.iterrows():
        chapter_tfidf = dict(zip(feature_names, tfidf_matrix[i].toarray()[0]))
        tfidf_term, tfidf_val = sorted(chapter_tfidf.items(), key=lambda item: item[1], reverse=True)[0]
        tfidf_results_chapters.append({
            'book': book,
            'chapter': row['chapter'],
            'tfidf_term': tfidf_term,
            'tfidf_val': tfidf_val
        })
    print(f'{book} done.')

tfidf_df_chapters = pd.DataFrame(tfidf_results_chapters)
tfidf_df_chapters.to_csv('../datasets/kjv_tfidf.csv', index=False)
print('kjv_tfidf.csv exported')

book_texts = df.groupby('book')['tokens'].apply(lambda x: ' '.join(x)).reset_index()
vectorizer = TfidfVectorizer()
tfidf_matrix_books = vectorizer.fit_transform(book_texts['tokens'])

tfidf_results_books = []
feature_names = vectorizer.get_feature_names_out()
for i, row in book_texts.iterrows():
    book_tfidf = dict(zip(feature_names, tfidf_matrix_books[i].toarray()[0]))
    tfidf_term, tfidf_val = sorted(book_tfidf.items(), key=lambda item: item[1], reverse=True)[0]
    tfidf_results_books.append({
        'book': row['book'],
        'tfidf_term': tfidf_term,
        'tfidf_val': tfidf_val
    })
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
tfidf_df_books = pd.DataFrame(tfidf_results_books)
tfidf_df_books['book'] = pd.Categorical(tfidf_df_books['book'], categories=bible_books_order, ordered=True)
tfidf_df_books = tfidf_df_books.sort_values('book')
tfidf_df_books.to_csv('../datasets/kjv_tfidf_books.csv', index=False)
print('kjv_tfidf_books.csv exported')