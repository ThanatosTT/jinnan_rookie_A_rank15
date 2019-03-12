#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# 时间格式处理函数
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

# 数据加载
def load_data(data_path):
    train_ = pd.read_csv(data_path + 'jinnan_round1_train_20181227.csv', encoding='gbk')
    optimize_ = pd.read_csv(data_path + 'optimize.csv', encoding='gbk')
    FuSai_ = pd.read_csv(data_path + 'FuSai.csv', encoding='gbk')

    train_['样本id'] = train_['样本id'].apply(lambda x: int(x.split('_')[1]))
    optimize_['样本id'] = optimize_['样本id'].apply(lambda x: int(x.split('_')[1]))
    FuSai_['样本id'] = FuSai_['样本id'].apply(lambda x: int(x.split('_')[1]))
    # 清洗训练集里的脏数据
    train_['A25'] = train_['A25'].apply(lambda x: int(x) if len(x) == 2 else np.nan)
    train_.loc[train_['样本id'] == 101, 'B2'] = 3.5
    train_.loc[train_['样本id'] == 101, 'B3'] = 3.5
    train_.loc[train_['样本id'] == 101, 'B1'] = 320
    train_.loc[train_['样本id'] == 102, 'B1'] = 320
    train_.loc[train_['B14'] == 40, 'B14'] = 400
    train_['A22'][train_['A22'] == 3.5] = 9
    train_['A23'][train_['A23'] == 10] = 5

    train_.loc[train_['样本id'] == 223, 'A5'] = '15:00:00'
    train_.loc[train_['样本id'] == 1023, 'A5'] = '21:30:00'
    train_.loc[train_['样本id'] == 1027, 'A5'] = '21:30:00'
    train_.loc[train_['样本id'] == 937, 'A9'] = '23:00:00'
    train_.loc[train_['样本id'] == 496, 'A9'] = '6:30:00'
    train_.loc[train_['样本id'] == 130, 'A11'] = '00:30:00'
    train_.loc[train_['样本id'] == 1067, 'A11'] = '21:30:00'
    train_.loc[train_['样本id'] == 933, 'A16'] = '12:00:00'
    train_.loc[train_['样本id'] == 1577, 'A24'] = '03:00:00'
    train_.loc[train_['样本id'] == 1577, 'A26'] = '03:30:00'
    train_.loc[train_['样本id'] == 534, 'A26'] = '19:30:00'
    train_.loc[train_['样本id'] == 1350, 'A26'] = '13:00:00'
    train_.loc[train_['样本id'] == 12, 'B5'] = '14:00:00'
    train_.loc[train_['样本id'] == 960, 'A20'] = '18:30-19:00'
    train_.loc[train_['样本id'] == 1537, 'A28'] = '15:40-16:10'
    train_.loc[train_['样本id'] == 1157, 'B4'] = '16:00-17:00'
    train_.loc[train_['样本id'] == 924, 'B10'] = '19:00-21:00'
    train_.loc[train_['样本id'] == 643, 'B11'] = '12:00-13:00'
    train_.loc[train_['样本id'] == 609, 'B11'] = '10:30-11:30'
    train_.loc[train_['样本id'] == 1164, 'B11'] = '4:00-5:00'
    train_ = train_[train_['收率'] > 0.85]
    return train_, optimize_, FuSai_


def feat_gen_step1(train_, optimize_, FuSai_):
    # 数据缺失值填充
    corr_columns = []
    fillna_columns = ['A2', 'A3', 'A7', 'A8', 'A21', 'A23', 'A25', 'A27', 'B1', 'B2', 'B3', 'B8', 'B10', 'B11',
                      'B12']  # 部分手动填充完毕
    for f in fillna_columns:
        for df in [train_, optimize_, FuSai_]:
            if f in ['A2', 'A3']:
                df[f] = df[f].fillna(0)
            elif f in ['A21', 'A23', 'A25', 'A27', 'B1', 'B2', 'B3', 'B8', 'B12']:
                df[f] = df[f].fillna(df[f].median())
            else:
                continue

    # 部分特征构造
    for df in [train_, optimize_, FuSai_]:
        df['A2 + A3 + A4'] = df['A2'] + df['A3'] + df['A4']

    for f in ['A2 + A3 + A4', 'A1', 'A2', 'A3', 'A4', 'A19', 'A21', 'B1', 'B12']:
        corr_columns.append(f + '+B14')
        corr_columns.append(f + '/B14')
        for df in [train_, optimize_, FuSai_]:
            df[f + '+B14'] = df[f] + df['B14']
            df[f + '/B14'] = df[f] / df['B14']

        train_ = pd.merge(train_, train_.groupby([f + '/B14'])['收率'].median().reset_index().rename(
            columns={'收率': f + '/B14' + '_rate_median'}), on=f + '/B14', how='left')
        optimize_ = pd.merge(optimize_, train_.groupby([f + '/B14'])['收率'].median().reset_index().rename(
            columns={'收率': f + '/B14' + '_rate_median'}), on=f + '/B14', how='left')
        FuSai_ = pd.merge(FuSai_, train_.groupby([f + '/B14'])['收率'].median().reset_index().rename(
            columns={'收率': f + '/B14' + '_rate_median'}), on=f + '/B14', how='left')

    data_optimize = pd.concat([train_, optimize_], axis=0, ignore_index=True)
    data_FuSai = pd.concat([train_, FuSai_], axis=0, ignore_index=True)
    return data_optimize, data_FuSai, corr_columns


def feat_gen_step2(data_optimize_, data_FuSai_):
    one_time_columns = ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7']
    time_per_columns = ['A20', 'A28', 'B4', 'B9', 'B10', 'B11']
    for column in one_time_columns:
        data_optimize_[column] = data_optimize_[column].apply(change_t)
        data_FuSai_[column] = data_FuSai_[column].apply(change_t)

    # 同操作间隔
    for column in time_per_columns:
        data_optimize_[column + '_diff'] = (data_optimize_[column].apply(lambda x: split_t(x, 1)) -
                                            data_optimize_[column].apply(lambda x: split_t(x, 0))).apply(judge_time)
        data_optimize_ = data_optimize_.drop([column], axis=1)

        data_FuSai_[column + '_diff'] = (data_FuSai_[column].apply(lambda x: split_t(x, 1)) -
                                         data_FuSai_[column].apply(lambda x: split_t(x, 0))).apply(judge_time)
        data_FuSai_ = data_FuSai_.drop([column], axis=1)
    return data_optimize_, data_FuSai_

# 时间缺失填充函数
def fill_time_nan(data):
    columns = ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7']
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

# 收率特征构造
def gen_feat_step3(data, train_):
    cate_columns = [f for f in data.columns if f != '样本id']
    for f in cate_columns:
        data[f] = data[f].map(dict(zip(data[f].unique(), range(0, data[f].nunique()))))
    train_data = data[:train_.shape[0]]
    test_data = data[train_.shape[0]:]

    le_train = train_data.copy()
    le_train['target'] = train['收率']
    le_train['intTarget'] = pd.cut(le_train['target'], 5, labels=False)
    le_train = pd.get_dummies(le_train, columns=['intTarget'])
    li = ['intTarget_0.0', 'intTarget_1.0', 'intTarget_2.0', 'intTarget_3.0', 'intTarget_4.0']

    mean_columns = ['A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A14', 'A15', 'A16', 'A17',
                    'A19', 'A21', 'A22', 'A24', 'A25', 'A26', 'A27', 'B1', 'B5', 'B6', 'B7', 'B8',
                    'B12', 'B14', 'A20_diff', 'A28_diff', 'B4_diff', 'B9_diff', 'B10_diff', 'B11_diff']
    for f1 in mean_columns:
        for f2 in li:
            col_name = f1 + "_" + f2 + '_mean'
            order_label = le_train.groupby([f1])[f2].mean()
            for df in [train_data, test_data]:
                df[col_name] = df['B14'].map(order_label)
    return train_data, test_data

# 丢弃部分特征
def drop_feat(train_, test_, corr_columns):
    drop_columns = ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7',
                    'A8', 'A10', 'A12', 'A15', 'A17', 'A25', 'A27', 'B6', 'B8'] + \
                   ['A1', 'A2', 'A3', 'A4', 'A13', 'A18', 'A23', 'B2', 'B3', 'B13']
    train_.drop(drop_columns, axis=1, inplace=True)
    test_.drop(drop_columns, axis=1, inplace=True)

    corr_matrix = train_[corr_columns].corr()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    threshold = 0.99
    drop = []
    for column in upper.columns:
        if any(upper[column] >= threshold):
            drop.append(column)
    train_ = train_.drop(drop, axis=1)
    test_ = test_.drop(drop, axis=1)
    return train_, test_

# 模型以及模型融合
def modeling_cross_validation(params, X, y, nr_folds=5):
    oof_preds = np.zeros(X.shape[0])
    # Split data with kfold
    folds = KFold(n_splits=nr_folds, shuffle=False, random_state=4096)

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        print("fold n°{}".format(fold_ + 1))
        trn_data = lgb.Dataset(X[trn_idx], y[trn_idx])
        val_data = lgb.Dataset(X[val_idx], y[val_idx])

        num_round = 20000
        clf = lgb.train(params, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=1000,
                        early_stopping_rounds=100)
        oof_preds[val_idx] = clf.predict(X[val_idx], num_iteration=clf.best_iteration)

    score = mean_squared_error(oof_preds, train_label)

    return score / 2


def lgb_model(X_train, y_train, X_test):
    param = {'num_leaves': 120,
             'min_data_in_leaf': 20,
             'objective': 'regression',
             'max_depth': -1,
             'learning_rate': 0.01,
             "min_child_samples": 20,
             "boosting": "gbdt",
             "feature_fraction": 0.6,
             "bagging_freq": 1,
             "bagging_fraction": 0.9,
             "bagging_seed": 11,
             "metric": 'mse',
             "lambda_l1": 0.1,
             "verbosity": -1}
    folds = KFold(n_splits=9, shuffle=True, random_state=2018)
    oof_lgb = np.zeros(len(train))
    predictions_lgb = np.zeros(X_test.shape[0])

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        print("fold n°{}".format(fold_ + 1))
        trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
        val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])

        num_round = 10000
        clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=200,
                        early_stopping_rounds=100)
        oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)

        predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

    print("lgb CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, y_train)))

    return predictions_lgb, oof_lgb


def xgb_model(X_train, y_train, X_test):
    xgb_params = {'eta': 0.01, 'max_depth': 6, 'subsample': 0.9, 'colsample_bytree': 0.6,
                  'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': -1}

    folds = KFold(n_splits=9, shuffle=True, random_state=2018)
    oof_xgb = np.zeros(len(train))
    predictions_xgb = np.zeros(X_test.shape[0])

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        print("fold n°{}".format(fold_ + 1))
        trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx])
        val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])

        watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
        clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                        verbose_eval=100, params=xgb_params)
        oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
        predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits

    print("xgb CV score: {:<8.8f}".format(mean_squared_error(oof_xgb, y_train)))
    return predictions_xgb, oof_xgb


def stacking_model(oof_lgb, oof_xgb, predictions_lgb, predictions_xgb):
    train_stack = np.vstack([oof_lgb, oof_xgb]).transpose()
    test_stack = np.vstack([predictions_lgb, predictions_xgb]).transpose()

    folds_stack = RepeatedKFold(n_splits=9, n_repeats=2, random_state=4590)
    oof_stack = np.zeros(train_stack.shape[0])
    predictions = np.zeros(test_stack.shape[0])

    for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack, y_train_)):
        trn_data, trn_y = train_stack[trn_idx], y_train_[trn_idx]
        val_data, val_y = train_stack[val_idx], y_train_[val_idx]

        clf_3 = BayesianRidge()
        clf_3.fit(trn_data, trn_y)

        oof_stack[val_idx] = clf_3.predict(val_data)
        predictions += clf_3.predict(test_stack) / 18

    loss = mean_squared_error(y_train_, oof_stack)
    print('merge loss:', loss)
    return predictions


if __name__ == "__main__":
    path = 'data/'
    train, optimize, FuSai = load_data(path)
    train_label = train['收率']
    optimize, FuSai, corr_feats = feat_gen_step1(train, optimize, FuSai)
    optimize, FuSai = feat_gen_step2(optimize, FuSai)
    optimize = fill_time_nan(optimize)
    FuSai = fill_time_nan(FuSai)
    optimize = optimize.fillna(-1)
    FuSai_dataset = FuSai.fillna(-1)
    train4optimize, test4optimize = gen_feat_step3(optimize, train)
    train4FuSai, test4FuSai = gen_feat_step3(FuSai, train)
    train4optimize, test4optimize = drop_feat(train4optimize, test4optimize, corr_feats)
    train4FuSai, test4FuSai = drop_feat(train4FuSai, test4FuSai, corr_feats)
    # 丢弃部分特征值
    best_features = [f for f in train4optimize.columns if
                     f not in ['样本id', '收率', 'B1', 'B1+B14', 'B12+B14', 'A28_diff', 'A6_intTarget_1.0_mean']]
    X_train_ = train4optimize[best_features].values
    X_test_optimize = test4optimize[best_features].values
    X_test_FuSai = test4FuSai[best_features].values
    y_train_ = train_label.values

    predictions_lgb_optimize_, oof_lgb_optimize_ = lgb_model(X_train_, y_train_, X_test_optimize)
    predictions_lgb_FuSai_, oof_lgb_FuSai_ = lgb_model(X_train_, y_train_, X_test_FuSai)

    predictions_xgb_optimize_, oof_xgb_optimize_ = xgb_model(X_train_, y_train_, X_test_optimize)
    predictions_xgb_FuSai_, oof_xgb_FuSai_ = xgb_model(X_train_, y_train_, X_test_FuSai)

    predictions_optimize = stacking_model(oof_lgb_optimize_, oof_xgb_optimize_, predictions_lgb_optimize_,
                                          predictions_xgb_optimize_)
    predictions_FuSai = stacking_model(oof_lgb_FuSai_, oof_xgb_FuSai_, predictions_lgb_FuSai_, predictions_xgb_FuSai_)

    sub_df_FuSai = pd.read_csv(path + 'FuSai.csv', encoding='gbk')
    sub_df_FuSai['收率'] = predictions_FuSai
    sub_df_FuSai['收率'] = sub_df_FuSai['收率'].apply(lambda x: round(x, 3))
    sub_df_FuSai[['样本id', '收率']].to_csv('submit_FuSai.csv', index=False, header=None, encoding='gbk')
    predictions_optimize = pd.DataFrame(predictions_optimize)
    predictions_optimize.columns = ['收率']
    predictions_optimize.to_csv('submit_optimize.csv', index=False)
