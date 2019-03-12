le_train['target'] = target
le_train['intTarget'] = pd.cut(le_train['target'], 5, labels=False)
train = pd.get_dummies(le_train, columns=['intTarget'])
li = ['intTarget_0.0', 'intTarget_1.0', 'intTarget_2.0', 'intTarget_3.0', 'intTarget_4.0']
mean_features = []

mean_columns = ['A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A14', 'A15', 'A16', 'A17',
                'A19', 'A21', 'A22', 'A24', 'A25', 'A26', 'A27', 'B1', 'B5', 'B6', 'B7', 'B8',
                'B12', 'B14', 'A20_diff', 'A28_diff', 'B4_diff', 'B9_diff', 'B10_diff', 'B11_diff']
for f1 in mean_columns:
    mean_features.append(f1)
    for f2 in li:
        col_name = f1 + "_" + f2 + '_mean'
        order_label = le_train.groupby([f1])[f2].mean()
        for df in [train, test]:
            df[col_name] = df['B14'].map(order_label)





