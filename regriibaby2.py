#import the libraries
import pandas as pd
import quandl
import math, datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, train_test_split

import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

#import the google stock data from quandle
#simply visit the quandl website and search for googl stock
#then copy the code there... which is used below or dowload the file
df=quandl.get('WIKI/GOOGL')

#check the first 5 columns to be sure you downloaded what you wanted
print(df.head())

#use only the useful columns
df=df[['Adj. Open', 'Adj. High','Adj. Low', 'Adj. Close','Adj. Volume']]

#percent volatility...percent change...high minus low percent
df['HL_PCT']=(df['Adj. High']-df['Adj. Close']) /df['Adj. Close']*100.0

#percentage change daily
df['PCT_Change']=(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100.0

#define a new dataframe with selected attributes
df=df[['Adj. Close','HL_PCT','PCT_Change','Adj. Volume']]
print(df.head())

#forecasting the adjusted close....forcasting deals with a single feature or column
forecast_col='Adj. Close'

#replace missing data...fill 'not available' with any chosen value
df.fillna(-9999,inplace=True)

#using regression to forecast out
#math.ceil roumds UP to nearest whole..then int makes it int
forecast_out=int(math.ceil(0.01*len(df)))

#creating the label and shifting the col negatively...predicting 10% out
#t
df['label']=df[forecast_col].shift(-forecast_out)
print(df.head())

#exclude the label column and assign the rest columns to x
x=np.array(df.drop(['label'],1))

#preprocess the data and assign to range between -1 and 1
x=preprocessing.scale(x)

x=x[:-forecast_out]

x_lately=x[-forecast_out:]


#drop off all empty rows
df.dropna(inplace=True)


#assign the label column to y
y=np.array(df['label'])

#assign the label column to y
y=np.array(df['label'])

#splitting the dataset into training and testing... testing 20 percent or 0.2 out of 1
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2)

#call the regression algorithm to build the model with the train dataset
#assign the called algortihm to clf...or whatever you want
clf=LinearRegression(n_jobs=-1)

#build the model... the "fit" builds the model from the train dataset
clf.fit(x_train, y_train)


with open('linearregression.pickle','web') as f:
    pickle.dump(clf,f)
    
pickle_in=open('linearregression.pickle','rb')
clf=pickle.load(pickle_in)

#check the accuracy of the model
accuracy=clf.score(x_test, y_test)
print (accuracy)
                                                                                                                                       

forecast_set=clf.predict(x_lately)

print(forecast_set, accuracy, forecast_out)

df['Forecast']=np.nan

last_date=df.iloc[-1].name
last_unix=last_date.timestamp()
one_day=86400
next_unix=last_unix+one_day

for i in forecast_set:
    next_date=datetime.datetime.fromtimestamp(next_unix)
    next_unix+=one_day
    df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)]+[i]
    
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()







