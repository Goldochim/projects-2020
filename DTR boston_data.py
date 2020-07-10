# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 15:41:43 2020

@author: Gold
"""
from sklearn import datasets
from sklearn.tree import  DecisionTreeRegressor
from sklearn import tree
from matplotlib import pyplot as plt

boston=datasets.load_boston()
x=boston.data
y=boston.target

regr=DecisionTreeRegressor(random_state=1234)
model=regr.fit(x,y)

text_rep=tree.export_text(regr)
print(text_rep)

accuracy=regr.score(x, y)
print (accuracy)

fig=plt.figure(figsize=(25,20))
graph=tree.plot_tree(regr, feature_names=boston.feature_names, filled=True)

#To save the image to png
fig.savefig("decision_tree_regressor.png")