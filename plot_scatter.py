import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold, RepeatedKFold
import warnings
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', 100)

train = pd.read_csv('original_data/train.csv', encoding='gbk')
test = pd.read_csv('original_data/test.csv', encoding='gbk')

# 数据清洗 预处理
train['样本id'] = train['样本id'].apply(lambda x: int(x.split('_')[1]))
test['样本id'] = test['样本id'].apply(lambda x: int(x.split('_')[1]))
train['A25'] = train['A25'].apply(lambda x: int(x) if len(x) == 2 else np.nan)
train.loc[train['样本id'] == 101, 'B2'] = 3.5
train.loc[train['样本id'] == 101, 'B3'] = 3.5
train.loc[train['样本id'] == 101, 'B1'] = 320
train.loc[train['样本id'] == 102, 'B1'] = 320
train.loc[train['B14'] == 40, 'B14'] = 400

train['A22'][train['A22'] == 3.5] = 9
train['A23'][train['A23'] == 10] = 5

train.loc[train['样本id'] == 223, 'A5'] = '15:00:00'
train.loc[train['样本id'] == 1023, 'A5'] = '21:30:00'
train.loc[train['样本id'] == 1027, 'A5'] = '21:30:00'

train.loc[train['样本id'] == 937, 'A9'] = '23:00:00'
train.loc[train['样本id'] == 496, 'A9'] = '6:30:00'

train.loc[train['样本id'] == 130, 'A11'] = '00:30:00'
train.loc[train['样本id'] == 1067, 'A11'] = '21:30:00'

train.loc[train['样本id'] == 933, 'A16'] = '12:00:00'
train.loc[train['样本id'] == 1350, 'A26'] = '13:00:00'
train.loc[train['样本id'] == 969, 'B4'] = '19:00-20:05'
train.loc[train['样本id'] == 1106, 'B4'] = '15:00-16:00'
train.loc[train['样本id'] == 12, 'B5'] = '14:00:00'

train.loc[train['样本id'] == 1577, 'A24'] = '03:00:00'
train.loc[train['样本id'] == 1577, 'A26'] = '03:30:00'
train.loc[train['样本id'] == 534, 'A26'] = '19:30:00'
test.loc[test['B14'] == 785, 'B14'] = 385
test.loc[test['样本id'] == 123, 'A25'] = 71
test.loc[test['样本id'] == 123, 'A27'] = 76
test.loc[test['样本id'] == 23, 'A12'] = 102
test.loc[test['样本id'] == 54, 'A19'] = 300
test.loc[test['样本id'] == 966, 'A19'] = 300
test.loc[test['样本id'] == 761, 'A20'] = '22:30-23:00'
test.loc[test['样本id'] == 1229, 'A5'] = '23:00:00'

# 删除类别唯一的特征 90%的列
for df in [train, test]:
    df.drop(['A1', 'A2', 'A3', 'A4', 'A13', 'A18', 'A23', 'B2', 'B3', 'B13'], axis=1, inplace=True)
# 材料填充　温度填充
material_columns = ['A19', 'A21', 'B1', 'B12', 'B14']  # 9　'A1', 'A2', 'A3', 'A4'删
temperature_columns = ['A6', 'A8', 'A10', 'A12', 'A15', 'A17', 'A25', 'A27', 'B6', 'B8']  # 10
# concentration_columns = ['B2', 'B13'] 删
ph_columns = ['A22']  # 'A23','B3' 删
# pressure_columns = ['A13', 'A18'] 删

for f in material_columns + temperature_columns:
    for df in [train, test]:
        train[f] = train[f].fillna(train[f].median())

# 收率特征
train = train[train['收率'] > 0.87]
for f in material_columns + ph_columns + ['A6']:
    for df in [train, test]:
        # df[f'{f}_mean_sl'] = df[f].map(train.groupby([f])['收率'].mean())
        df[f'{f}_median_sl'] = df[f].map(train.groupby([f])['收率'].median())
    if f != 'B1收率4':
        for df in [train, test]:
            df[f'{f}/B14'] = df[f] / df['B14']
            # df[f'{f}/B14_mean_sl'] = df[f'{f}/B14'].map(train.groupby([f'{f}/B14'])['收率'].mean())
            df[f'{f}/B14_median_sl'] = df[f'{f}/B14'].map(train.groupby([f'{f}/B14'])['收率'].median())
train_label = train['收率']
del train['收率']
data = pd.concat([train, test], axis=0, ignore_index=True)

# 操作时间列 错误由７３减少到１２个
one_time_columns = ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7']  # 10
time_per_columns = ['A20', 'A28', 'B4', 'B9', 'B10', 'B11']  # 6
error = []


def change_t(x):
    try:
        t = int(x.split(':')[0]) + int(x.split(':')[1]) / 60
    except:
        if isinstance(x, str):
            if ';' in x:
                return int(x.split(';')[0]) + int(x.split(';')[1]) / 60
            elif '；' in x:
                return int(x.split('；')[0]) + int(x.split('；')[1]) / 60
            elif '::' in x:
                return int(x.split('::')[0]) + int(x.split('::')[1]) / 60
            elif '\"' in x:
                return int(x.split('\"')[0]) + int(x.split('\"')[1]) / 60
            elif '分' in x:
                return int(x[:-1].split(':')[0]) + int(x[:-1].split(':')[1]) / 60
            elif x == '19:':
                return 19
            elif x == '1600':
                return 16
            else:
                error.append(x)
                return np.nan
        else:
            return np.nan
    else:
        return t


def split_t(x, i):
    try:
        x = x.split('-')[i]
        t = change_t(x)
    except:
        return np.nan
    else:
        return t


def judge_time(x):
    if x >= 0:
        return x
    else:
        return x + 24


# 时间格式转换
for column in one_time_columns:
    data[column] = data[column].apply(change_t)

# 同操作间隔
for column in time_per_columns:
    data[column + '_first'] = data[column].apply(lambda x: split_t(x, 0))
    data[column + '_second'] = data[column].apply(lambda x: split_t(x, 1))
    data[column + '_diff'] = (data[column + '_second'] - data[column + '_first']).apply(judge_time)
    data = data.drop([column], axis=1)
    del data[column + '_first']
    del data[column + '_second']  # 留着效果不好

# time缺失值填充
time_columns = one_time_columns[:]


def fill_time_nan(data, columns):
    for i in range(len(columns) - 1):
        column_i = columns[i]
        column_j = columns[i + 1]
        diff_time = (data[column_j] - data[column_i]).apply(judge_time).median()
        data[column_i] = data[column_i].fillna(data[column_j] - diff_time)
    for i in range(len(columns) - 1, 0, -1):
        column_i = columns[i]
        column_j = columns[i - 1]
        diff_time = (data[column_i] - data[column_j]).apply(judge_time).median()
        data[column_i] = data[column_i].fillna(data[column_j] + diff_time)
    return data


data = fill_time_nan(data, time_columns)
data = data.fillna(-1)
train = data[:train.shape[0]]
test = data[train.shape[0]:]
train['收率'] = list(train_label)
for f in ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7']:
    train.plot.scatter(x='样本id', y='收率')
    plt.title("收率")
    plt.xlabel(f, fontsize=12)
    plt.xlim()
    plt.ylim(0.7,1.1)
    plt.axis()
    plt.show()
