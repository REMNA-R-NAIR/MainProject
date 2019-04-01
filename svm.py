import pandas as pd
import numpy as np
from sklearn import metrics

data=pd.read_csv("feature.csv")
data_new=pd.read_csv("feature.csv",na_values=['?'])

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
print(y_train)


# Applying SVM 

import sklearn
from sklearn import svm

C = 1.0
svc = svm.SVC(kernel='linear',C=C,gamma=2)
svc.fit(X_train, y_train)


print ("xtest",X_test)
print("ytest",y_test)
predictions_test = svc.predict(X_test.iloc[[2]])
print("predictions",predictions_test)
#print("Accuracy is",metrics.accuracy_score(y_test,predictions_test))






