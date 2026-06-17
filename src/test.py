import pandas as pd

df = pd.read_parquet("data/gold/Article_ai_ready.parquet")

colonnes_a_verifier = [
    'category_l1', 
    'category_l2', 
    'category_l3', 
    'category_l4', 
    'category_l5'
]

df_filtre = df[~df[colonnes_a_verifier].eq('Non classé').all(axis=1)]

for idx, row in df_filtre.head(n=500).iterrows():
    print('panel: ',row['panel_clean'])
    print('designation: ', row['designation_clean'])
    # print('category 1: ', row['category_l1'])
    # print('category 2: ', row['category_l2'])
    # print('category 3: ', row['category_l3'])
    # print('category 4: ', row['category_l4'])
    # print('category 5: ', row['category_l5'])
    print('-'*60)