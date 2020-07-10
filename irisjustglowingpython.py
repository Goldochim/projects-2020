# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 14:48:53 2020

@author: Gold
"""

from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
#from dtreeviz.trees import dtreeviz

iris=datasets.load_iris()
x=iris.data
y=iris.target

print(x[:9])
print(y[17:30])

clf=DecisionTreeClassifier(random_state=1234)
clf.fit(x,y)
#print (clf)
text_rep=tree.export_text(clf)
#print(text_rep)

#to save the text_rep to text, do the following
#with open("decision_tree.log","w") as fout:
#    fout.write(text_rep)

#To plot the tree
fig=plt.figure(figsize=(25,20))
graph=tree.plot_tree(clf,feature_names=iris.feature_names, class_names=iris.target_names, filled=True)

#To save the image to png
#fig.savefig("decision_tree.png")


#using graphviz to visualize the tree
#dot_dat=tree.export_graphviz(clf,out_file=None,feature_names=iris.feature_names,class_names=iris.target_names, filled=True)
#graph=graphviz.Source(dot_dat, format="png")
#graph

#plotting using dtreeviz
#viz=dtreeviz(clf,x,y,target_names="target", feature_names=iris.feature_names, class_names=list(iris.target_names))
#viz

accuracy=clf.score(x, y)
print (accuracy)




    