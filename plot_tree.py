import xgboost as xgb
from sklearn.datasets import load_iris
iris = load_iris()

xgb_clf = xgb.XGBClassifier()
xgb_clf.fit(iris.data, iris.target)
digraph = xgb.to_graphviz(xgb_clf, num_trees=1)
digraph.format = 'png'
digraph.view('./iris_xgb')
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10, 10))
ax = fig.subplots()
xgb.plot_tree(xgb_clf, num_trees=1, ax=ax)
plt.show()
