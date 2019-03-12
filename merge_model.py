import pandas as pd
import os

path = 'stacking_model/'
name_list = [i for i in os.listdir(path)]
i = 0
for name in name_list:
    if i == 0:
        sub_df = pd.read_csv(path + name, header=None)
        sub_df[1] = sub_df[1]/30
    else:
        sub = pd.read_csv(path + name, header=None)
        sub_df[1] += sub[1] / 30
    i += 1
sub_df[1] = sub_df[1].apply(lambda x: round(x, 3))
sub_df.to_csv('sub/merge_model.csv', index=False, header=False)
