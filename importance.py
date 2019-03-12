import operator
import matplotlib.pyplot as plt
import pandas as pd


def ceate_feature_map(features):
    outfile = open('map_dic/xgb.fmap', 'w', encoding='utf-8')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()


def plt_importance(clf, feature_columns):
    features = list(feature_columns)
    ceate_feature_map(features)
    importance = clf.get_fscore(fmap='map_dic/xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(15, 15))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    plt.show()
