from thefuzz import fuzz
import pandas as pd


df = pd.read_csv('../datasets/kjv.csv')

dfs = [df[df['book'] == book] for book in df['book'].unique()]

methods = [
    fuzz.token_set_ratio, 
    fuzz.token_sort_ratio, 
    fuzz.partial_token_set_ratio, 
    fuzz.partial_token_sort_ratio
]

print('Computing similarities...')

for m in range(len(methods)):
    df_data = {
        'book_1': list(),
        'book_2': list(),
        'comparisons': list(),
        'similarity_mean': list()
    }
    print(f'Method: {methods[m].__name__}')
    for i in range(len(dfs) - 1):
        for j in range(i + 1, len(dfs)):
            similarity = 0
            comparisons = 0
            for k in range(dfs[i].shape[0]):
                vers_1 = dfs[i].iloc[k]['text'].replace('\n', '')
                for l in range(dfs[j].shape[0]):
                    vers_2 = dfs[j].iloc[l]['text'].replace('\n', '')
                    similarity += methods[m](vers_1, vers_2)
                    comparisons += 1
            print(f'{dfs[i].iloc[0]["book"]} ----> {dfs[j].iloc[0]["book"]} done.')
            df_data['book_1'].append(dfs[i].iloc[0]['book'])
            df_data['book_2'].append(dfs[j].iloc[0]['book'])
            df_data['comparisons'].append(comparisons)
            df_data['similarity_mean'].append(round(similarity / comparisons, 2))
    pd.DataFrame(df_data).to_csv(f'../datasets/books_fuzzy_comparative_analysis_{methods[m].__name__}.csv', index=False)
    print(f'Exported books_fuzzy_comparative_analysis_{methods[m].__name__}.csv')
