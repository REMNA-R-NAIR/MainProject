#Decision tree Classifier

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

data=pd.read_csv("ganesha.csv")
data_new=pd.read_csv("ganesha.csv",na_values=['?'])

data_new.dropna(inplace=True)
predictions=data_new['gender']
data_new
features_raw = data_new[['nobs','mean','skew','kurtosis','median','mode','std','low','peak','q25','q75','iqr']]
#test_raw=test_new[['nobs','mean','skew','kurtosis','median','mode','std','low','peak','q25','q75','iqr']]
from sklearn.model_selection import train_test_split

predict_class = predictions.apply(lambda x: 0 if x == "Male" else 1)
np.random.seed(1234)

X_train, X_test, y_train, y_test = train_test_split(features_raw, predict_class, train_size=0.50, random_state=1)
# Show the results of the split

print ("Training set has {} samples.".format(X_train.shape[0]))
print ("Testing set has {} samples.".format(X_test.shape[0]))
#print(y_train)

tree = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
print("Decision Tree")
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

