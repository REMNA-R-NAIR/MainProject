                                                        #Random Forest

                                                      ####Testing###########
###############################################################################################################################################
#Random Forest
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft
import numpy as np
import pandas as pd
import re
import scipy.stats as stats
from scipy.io import wavfile
import numpy as np
import os


#rate,data=wavfile.read('/home/user/test/b0476.wav')
rate,data=wavfile.read('/home/user/test/a0044.wav')
        #get dominating frequencies in sliding windows of 200ms
step = rate//5 #3200 sampling points every 1/5 sec 
window_frequencies = []
for i in range(0,len(data),step):
    ft = np.fft.fft(data[i:i+step])
    freqs = np.fft.fftfreq(len(ft)) #fftq tells you the frequencies associated with the coefficients
    imax = np.argmax(np.abs(ft))
    freq = freqs[imax]
    freq_in_hz = abs(freq *rate)
    window_frequencies.append(freq_in_hz)
    filtered_frequencies = [f for f in window_frequencies if 20<f<300 and not 46<f<66] 
#print(filtered_frequencies)
nobs, minmax, mean, variance, skew, kurtosis = stats.describe(filtered_frequencies)
median    = np.median(filtered_frequencies)
mode      = stats.mode(filtered_frequencies).mode[0]
std       = np.std(filtered_frequencies)
low,peak  = minmax
q75,q25   = np.percentile(freqs, [75 ,25])
iqr       = q75 - q25

#print(nobs)
#print(minmax)
#print(mean)
#print(variance)
#print(skew) 
#print(kurtosis)
#print(median)
#print(mode)
#print(std)
#print(low)
#print(peak)
#print(q75)
#print(q25)
#print(iqr)
columns=['nobs', 'mean', 'skew', 'kurtosis', 
         'median', 'mode', 'std', 'low', 
         'peak', 'q25', 'q75', 'iqr', 
         'user_name', 'sample_date', 'age_range', 
        'pronunciation', 'gender' ]
myTest = pd.DataFrame(columns=columns)
sample_dict = {'nobs':nobs, 'mean':mean, 'skew':skew, 'kurtosis':kurtosis,
               'median':median, 'mode':mode, 'std':std, 'low': low,
               'peak':peak, 'q25':q25, 'q75':q75, 'iqr':iqr}
myTest.loc[1] = pd.Series(sample_dict)
myTest.to_csv('t.csv')




test_new=pd.read_csv("t.csv")

test_raw=test_new[['nobs','mean','skew','kurtosis','median','mode','std','low','peak','q25','q75','iqr']]
from sklearn.model_selection import train_test_split

predict_class = predictions.apply(lambda x: 0 if x == "Male" else 1)
np.random.seed(1234)

#X_train, X_test, y_train, y_test = train_test_split(features_raw, predict_class, train_size=0.80, random_state=1)

X_test=test_raw
# Show the results of the split

#print ("Training set has {} samples.".format(X_train.shape[0]))
#print ("Testing set has {} samples.".format(X_test.shape[0]))



# Applying desicion tree



C = 1.0
#svc = svm.SVC(kernel='linear',C=C,gamma=2)
#svc.fit(X_train, y_train)

from sklearn.metrics import fbeta_score
#print (X_test)
predictions_test = forest.predict(X_test)
predictions_test
#predictions_test
    #predictions_test

if(predictions_test[0]==1):
    print("Male")
else:
    print("Female")

    




    



