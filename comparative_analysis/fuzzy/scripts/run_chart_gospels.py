import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd


dts = [
    'gospel_fuzzy_comparative_analysis_partial_token_set_ratio.csv',
    'gospel_fuzzy_comparative_analysis_partial_token_sort_ratio.csv',
    'gospel_fuzzy_comparative_analysis_token_set_ratio.csv',
    'gospel_fuzzy_comparative_analysis_token_sort_ratio.csv'
]

for dt in dts:
    df = pd.read_csv('../datasets/' + dt)
    pivot_table = pd.pivot_table(df, values='similarity_mean', index='book_1', columns='book_2')
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, cmap='Blues', fmt='.2f', cbar_kws={'label': 'Similarity (%)'})
    plt.xlabel('')
    plt.ylabel('')
    plt.title(f'Heatmap - Similarity between Gospels with {dt.split('_analysis_')[-1].replace('.csv', '').replace('_', ' ').title()}')
    plt.savefig(f'../charts/similarity_gospels_heatmap_{dt.split("_analysis_")[-1].replace(".csv", "")}.png')
    print(f'Exported chart similarity_gospels_heatmap_{dt.split("_analysis_")[-1].replace(".csv", "")}.png')
