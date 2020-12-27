import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

#importing the dataset and reading it
df=pd.read_csv('spambase.data')
#print(df.head)
print(df.describe())

#creating the feature set by dropping the class column
x=np.array(df.drop(['spam_non_spam_class'], 1))
#preprocessing
x=preprocessing.scale(x)
y=np.array(df['spam_non_spam_class'])
#splitting the dataset to test and train dataset
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)

#runnig the support vector classifier algorithm on the dataset
svm_clf=SVC(gamma='auto', C=10)
svm_clf=svm_clf.fit(x_train, y_train)

#runnig the logistic Regression classifier algorithm on the dataset
lr_clf=LogisticRegression(random_state=0, solver='newton-cg')
lr_clf=lr_clf.fit(x_train, y_train)

#runnig the K-Nearest Neighbor classifier algorithm on the dataset
knn_clf=KNeighborsClassifier(n_neighbors=11)
knn_clf=knn_clf.fit(x_train, y_train)

#creating the dataset for the confusion matrix for all three algorithms because the confusion matrix uses the result for the predicted classes and the actual test dataset 
#first line is the class of the given dataset(the one set aside for testing)
#The lines that follow are the predicted values from the classifier
y_act=y_test
y_pred_svm=svm_clf.predict(x_test)
y_pred_lr=lr_clf.predict(x_test)
y_pred_knn=knn_clf.predict(x_test)

#printing the confusion matrix and Accuracy for the corresponding algorithms
print('SVM Accuracy and Confusion Matrix: '.upper())
accuracy_svm_test=svm_clf.score(x_test,y_test)
accuracy_svm_train=svm_clf.score(x_train,y_train)
print("svm test acc: ", {accuracy_svm_test})
print("svm_train_acc", {accuracy_svm_train})
ConfusionMat_svm=metrics.confusion_matrix(y_act, y_pred_svm)
print(ConfusionMat_svm)
print (metrics.classification_report (y_act,y_pred_svm))
print('------------------------------------------------------')

print('Logistic Reg Accuracy and Confusion Matrix: '.upper())
accuracy_lr_test=lr_clf.score(x_test,y_test)
accuracy_lr_train=lr_clf.score(x_train,y_train)
print("lr test acc: ", {accuracy_lr_test})
print("lr train acc: ", {accuracy_lr_train})
ConfusionMat_lr=metrics.confusion_matrix(y_act, y_pred_lr)
print(ConfusionMat_lr)
print (metrics.classification_report (y_act,y_pred_lr))
print('------------------------------------------------------')

print('K-Neighbors Accuracy and Confusion Matrix: '.upper())
accuracy_knn_test=knn_clf.score(x_test,y_test)
accuracy_knn_train=knn_clf.score(x_train,y_train)
print("knn test acc: ", {accuracy_knn_test})
print("knn train acc: ", {accuracy_knn_train})
ConfusionMat_knn=metrics.confusion_matrix(y_act, y_pred_knn)
print(ConfusionMat_knn)
print (metrics.classification_report (y_act,y_pred_knn))

#Predictions
example_measures=np.array([0,0,0,0,3,0,0,2,0,0,1,2,0,1,0,1,0,1,7,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,4.3333, 5,13])
example_measures=example_measures.reshape(1, -1)
prediction_svm=svm_clf.predict(example_measures)
prediction_lr=lr_clf.predict(example_measures)
prediction_knn=knn_clf.predict(example_measures)
print(prediction_svm)
print(prediction_lr)
print(prediction_knn)

#horizontal bar plots of the test accuracies
ax=sns.barplot(x=[accuracy_svm_test,accuracy_lr_test,accuracy_knn_test], y=['accuracy_svm_test','accuracy_lr_test','accuracy_knn_test'], hue=[accuracy_svm_train, accuracy_lr_train,accuracy_knn_train])
fig=plt.figure()
fig.suptitle('Algorithm Comparison')
ax.set_xlabel('Test Accuracies')
plt.show()




#NOTE NOTE NOTE NOTE NOTE
#The spam dataset was gotten from kaggle website. Three algorithms were used. The support vector mahine(svm_clf), the Logistics Regression(lr_clf), and K-Nearest Neighbor(knn_clf). The data set was split into a ratio of 8:2. 80% of the dataset was to trian the model and 20% was for testing.... Acuracies of the models were made based on the trainng and testing dataset. The model was generalised as it fared very well when the test accuracy was compared to the training accuracy..
#The results of the models are as follows: 
#svm test acc:  {0.9478827361563518}
#svm_train_acc {0.9657608695652173}
#------------------------------------------------------
#lr test acc:  {0.9381107491856677}
#lr train acc:  {0.9301630434782608}
#------------------------------------------------------
#knn test acc:  {0.9044516829533116}
#knn train acc:  {0.9165760869565217}
#All three models performed very well but the Support Vetor machine was seen to have the highest accuracy followed by the Logistics Regression. The K-Nearest Neighbor was seen to have the lowest accuracy amongst the three models. Although its accuray was very good as well.
#The confusion matrix was also displayed to show how the algorithm fared in terms of individul numbers.
#classification report was also shown for all models to show more details about their performance.
#The model was later tested with a strutured data from a spam email. This was collected manually from the mail. The logistic regression  and K-Nearest Neighbor models predicted the data to be a spam but Support vector machine predited it to be non spam. This is quite understandable because spams mean differently to people/machines/E-mail Applications.
#The result of the single/individual prediiton is shown below for svm, lr and knn in that order.
#[0]
#[1]
#[1]
#It was generally great to see how this models fared. Although spams differ, there were general similarities that were shared in the dataset. This dataset should also shed more light to organisations on how to draft emails so that they are not classified as spams. Although a different model will help shed more light in that. Probably somehting like Decision tree.
#A bar graph was plotted to show the difference in the acuries of the model. The bar graph has most values close which goes straight to quickly showing that the models accuracies are quite close.