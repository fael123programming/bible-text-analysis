import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd


if __name__ == '__main__':
    file_path = '../datasets/kjv_tfidf.csv'
    df = pd.read_csv(file_path)
    df['chapter'] = df['chapter'].astype(str)
    output_dir = '../charts'
    os.makedirs(output_dir, exist_ok=True)
    for book in df['book'].unique():
        book_df = df[df['book'] == book]
        # chart_data = book_df.groupby(['book', 'chapter'])['sentiment_score'].mean().reset_index()
        # chart_data['chapter'] = pd.to_numeric(chart_data['chapter'], errors='coerce')
        # chart_data = chart_data.sort_values('chapter')
        is_40 = len(book_df['chapter'].values) >= 40
        if is_40:
            plt.figure(figsize=(40, 10), dpi=300)
        else:
            plt.figure(figsize=(12, 8))
        sns.scatterplot(data=book_df, x='chapter', y='tfidf_val', legend=None)
        # mx = max(chart_data['chapter']) if len(chart_data['chapter']) > 1 else 0.1
        # my = max(chart_data['sentiment_score']) if len(chart_data['chapter']) > 1 else 0.5
        is_multiple_chapters_book = len(book_df['chapter'].values) > 1
        min_score, max_score = round(min(book_df['tfidf_val']), 2), round(max(book_df['tfidf_val']), 2)
        for _, row in book_df.iterrows():
            plt.vlines(x=row['chapter'], ymin=0, ymax=row['tfidf_val'], color='gray', linestyle='--', alpha=0.3)
            color = 'black'
            if is_multiple_chapters_book:
                if round(row['tfidf_val'], 2) == min_score:
                    color = 'red'
                elif round(row['tfidf_val'], 2) == max_score:
                    color = 'green'
            plt.text(x=row['chapter'], y=row['tfidf_val'], s=row['tfidf_term'], color=color, fontsize=9)
            # plt.text(x=row['chapter'] - (0.014 * mx), y=row['sentiment_score'] + (0.008 * my), s=f"{row['sentiment_score']:.2f}", color=color, fontsize=9)
        plt.title(f'TF-IDF Scatter Plot - {book}')
        plt.xlabel('Chapter')
        plt.ylabel('TF-IDF Value')
        plt.xticks(rotation=45, ticks=book_df['chapter'])
        plt.yticks(rotation=0)
        chart_file = os.path.join(output_dir, f'{book.strip().replace(' ', '_').lower()}_scatter_plot.png')
        plt.savefig(chart_file)
        plt.close()
        print(f'Saved chart for {book} to {chart_file}')

    file_path_books = '../datasets/kjv_tfidf_books.csv'
    df = pd.read_csv(file_path_books)
    plt.figure(figsize=(40, 10))
    sns.scatterplot(data=df, x='book', y='tfidf_val', legend=None)
    min_score, max_score = round(min(df['tfidf_val']), 2), round(max(df['tfidf_val']), 2)
    for _, row in df.iterrows():
        plt.vlines(x=row['book'], ymin=0, ymax=row['tfidf_val'], color='gray', linestyle='--', alpha=0.3)
        color = 'black'
        if round(row['tfidf_val'], 2) == min_score:
            color = 'red'
        elif round(row['tfidf_val'], 2) == max_score:
            color = 'green'
        plt.text(x=row['book'], y=row['tfidf_val'], s=row['tfidf_term'], color=color, fontsize=9)
    plt.title(f'TF-IDF Scatter Plot - KJV')
    plt.xlabel('Book')
    plt.ylabel('TF-IDF Value')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    chart_file = os.path.join(output_dir, 'kjv_scatter_plot.png')
    plt.savefig(chart_file)
    plt.close()
    print(f'Saved chart of KJV books')

