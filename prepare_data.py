
import os
import pandas as pd

def clean_data(path):
    df = pd.read_csv(path, sep=';', header = 0)
    
    list_of_features = ['pre']
    metrics = ['FOUT', 'MLOC', 'NBD', 'PAR', 'VG', 'NOF', 'NOM', 'NSF', 'NSM', 'ACD', 'NOI', 'NOT', 'TLOC', 'NOCU']
    list_of_features.extend([x for x in df.columns.values if x.split('_')[0] in metrics])
    clean_df = df.loc[:, list_of_features]
    clean_df['post'] = df['post']
    clean_df.loc[clean_df['post'] > 0, 'post'] = 1
    
    if not os.path.exists('./data/cleaned'):
        os.makedirs('./data/cleaned')
        
    clean_df.to_csv(f'./data/cleaned/{path.split("/")[-1]}', index=False)

clean_data('./data/eclipse-metrics-packages-2.0.csv')
clean_data('./data/eclipse-metrics-packages-2.1.csv')
clean_data('./data/eclipse-metrics-packages-3.0.csv')