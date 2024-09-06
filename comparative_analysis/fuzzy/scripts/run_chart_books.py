import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from time import sleep


dts = [
    'books_fuzzy_comparative_analysis_partial_token_set_ratio.csv',
    'books_fuzzy_comparative_analysis_partial_token_sort_ratio.csv',
    'books_fuzzy_comparative_analysis_token_set_ratio.csv',
    'books_fuzzy_comparative_analysis_token_sort_ratio.csv'
]

for dt in dts:
    df = pd.read_csv('../datasets/' + dt)
    pivot_table = pd.pivot_table(df, values='similarity_mean', index='book_1', columns='book_2')
    biblical_order = [
        "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
        "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel",
        "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles", 
        "Ezra", "Nehemiah", "Esther", "Job", "Psalms", "Proverbs", 
        "Ecclesiastes", "Song of Solomon", "Isaiah", "Jeremiah", 
        "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel", 
        "Amos", "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", 
        "Zephaniah", "Haggai", "Zechariah", "Malachi", 
        "Matthew", "Mark", "Luke", "John", "Acts", "Romans", 
        "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians", 
        "Philippians", "Colossians", "1 Thessalonians", 
        "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus", 
        "Philemon", "Hebrews", "James", "1 Peter", "2 Peter", 
        "1 John", "2 John", "3 John", "Jude", "Revelation"
    ]
    pivot_table = pivot_table.reindex(index=biblical_order, columns=biblical_order)
    plt.figure(figsize=(64, 48))
    sns.heatmap(pivot_table, annot=True, cmap='Blues', fmt='.2f', cbar_kws={'label': 'Similarity (%)'}, linewidths=0.1, linecolor='gray')
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(rotation=45)
    plt.title(f'Heatmap - Similarity between Books with {dt.split('_analysis_')[-1].replace('.csv', '').replace('_', ' ').title()}')
    plt.savefig(f'../charts/similarity_books_heatmap_{dt.split("_analysis_")[-1].replace(".csv", "")}.png')
    print(f'Exported chart similarity_books_heatmap_{dt.split("_analysis_")[-1].replace(".csv", "")}.png')
