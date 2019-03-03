import cv2
import csv
import numpy as np
import pandas as pd

#reading the csv fie

data=pd.read_csv("voice.csv")
data_new=pd.read_csv("voice.csv",na_values=['?'])
data_new.dropna(inplace=True)
predictions=data_new['label']
data_new
features_raw = data_new[['meanfreq','sd','median','Q25','Q75','IQR','skew','kurt','sp.ent','sfm','mode','centroid','meanfun','minfun','maxfun','meandom','mindom','maxdom','dfrange','modindx']]

from sklearn.model_selection import train_test_split

# male=0 and female =1

predict_class = predictions.apply(lambda x: 0 if x == "male" else 1)
np.random.seed(1234)

#80% data used for training

X_train, X_test, y_train, y_test = train_test_split(features_raw, predict_class, train_size=0.80, random_state=1)


# Show the results of the split

print ("Training set has {} samples.".format(X_train.shape[0]))
print ("Testing set has {} samples.".format(X_test.shape[0]))

#svm training

import sklearn
from sklearn import svm

C = 1.0
svc = svm.SVC(kernel='linear',C=C,gamma=2)
svc.fit(X_train, y_train)

from sklearn.metrics import fbeta_score
#print (X_test)
predictions_test = svc.predict(X_test)
predictions_test




