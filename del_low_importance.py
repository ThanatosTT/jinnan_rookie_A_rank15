column_importance = pd.DataFrame({'fea': list(train.columns), 'imp': clf.feature_importance()})
remove_col = []
for idx in range(len(column_importance)):
    col = column_importance['fea'][idx]
    wgt = column_importance['imp'][idx]
    if wgt <= 0:
        if col in new_train_x.columns:
            remove_col.append(col)
            new_train_x = new_train_x.drop(col, axis=1)
            testB_x = testB_x.drop(col, axis=1)
print(new_train_x.shape), print(testB_x.shape)