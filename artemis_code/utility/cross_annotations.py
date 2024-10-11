import pandas as pd
import random
import os


def cross_annotations(k):
    path = ("/Users/edoardofazzari/Library/CloudStorage/OneDrive-ScuolaSuperioreSant'Anna/"
            "PhD/reseaches/thesis/DATASETS/animalkingdom/annotation/")

    df = pd.read_csv(os.path.join(path, 'train_light.csv'), delimiter=';')
    indexes = list(range(0, df.shape[0]))
    random.shuffle(indexes)
    for i in range(k):
        consider_indexes = indexes[len(indexes)//k * i: len(indexes)//k * (i+1)]
        rows = df.iloc[consider_indexes]
        rows.to_csv(os.path.join(path, f'{i}_light.csv'), index=False)


if __name__ == '__main__':
    cross_annotations(5)
