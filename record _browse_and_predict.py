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
import tkinter as tk

from tkinter import *
from tkinter import filedialog
#tkinter creation and defining window size
master = Tk()
master.geometry("500x500")
def callback():
    import pyaudio
    import wave

    CHUNK = 1024 
    FORMAT = pyaudio.paInt16 #paInt8
    CHANNELS = 2 
    RATE = 44100 #sample rate
    RECORD_SECONDS = 4
    WAVE_OUTPUT_FILENAME = "recordedvoice.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,

                frames_per_buffer=CHUNK) #buffer

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data) # 2 bytes(16 bits) per channel

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()
    
 

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
def browsefun():
    global filename
    master.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("audio","*.wav"),("all files","*.*")))
    print(master.filename)



def find():
   
    rate,data=wavfile.read('/home/user/project/voice-gender-classifier-master/recordedvoice.wav')
    #rate,data=wavfile.read('/home/user/test/b0476.wav')
    #get dominating frequencies in sliding windows of 200ms
    step = rate//5 #3200 sampling points every 1/5 sec 
    window_frequencies = []
    for i in range(0,len(data),step):
        ft = np.fft.fft(data[i:i+step])
        freqs = np.fft.fftfreq(len(ft)) #fftq tells you the frequencies associated with the coefficients
        p=np.abs(ft)
        #print(len(p))
        max=0
        for i in range(0,len(p)//2):
#             print(p[i][0])
            if(p[i][1]>=max):
               max=p[i][1]
               index=i
#     print("max",max,"indexx",i)           
    #imax = np.argmax(p)  
        imax = index
    #print(imax)    
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
      
    from tkinter import messagebox
    messagebox.showinfo("Features Extracetd","nobs :- "+str(nobs)+"\nminmax :- "+str(minmax)+"\nmean :- "+str(mean)+"\nvariance :- "+str(variance)+"\nskew :- "+str(skew)+"\nkurtosis :- "+str(kurtosis)+"\nmedian :- "+str(median)+"\nmode :- "+str(mode)+"\nstd :- "+str(std)+"\nlow :- "+str(low)+"\npeak :- "+str(peak)+"\nq25 :- "+str(q25)+"\nq75 :- "+str(q75)+"\niqr :- "+str(iqr))

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

    import sklearn
    from sklearn import svm

    C = 1.0
#svc = svm.SVC(kernel='linear',C=C,gamma=2)
#svc.fit(X_train, y_train)

    from sklearn.metrics import fbeta_score
#print (X_test)
    predictions_test = forest.predict(X_test)
    predictions_test
#predictions_test
    #predictions_test

    if(predictions_test):
        print("MALE")
    else:
        print("FEMALE")
        
def detect():

    rate,data=wavfile.read('/home/user/project/voice-gender-classifier-master/recordedvoice.wav')
    #get dominating frequencies in sliding windows of 200ms

    step = rate//5 #3200 sampling points every 1/5 sec 
    window_frequencies = []
    for i in range(0,len(data),step):
        ft = np.fft.fft(data[i:i+step])
        freqs = np.fft.fftfreq(len(ft)) #fftq tells you the frequencies associated with the coefficients
        p=np.abs(ft)
        #print(len(p))
        max=0
        for i in range(0,len(p)//2):
#             print(p[i][0])
            if(p[i][1]>=max):
               max=p[i][1]
               index=i
#     print("max",max,"indexx",i)           
    #imax = np.argmax(p)  
        imax = index
    #print(imax)    
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

    import sklearn
    from sklearn import svm

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
        var="Male"
    else:
        var="Female"
    from tkinter import messagebox
    messagebox.showinfo("GENDER DETECTED", "the voice is of a "+var)
   # if(predictions_test):
    #    outLabel = Label(master, text = "MALE", anchor = S ).grid(row=21,column=3)
    #else:
     #   outLabel = Label(master, text = "FEMALE", anchor = S).grid(row=21,column=3)  

def browsefind():
    rate,data=wavfile.read(master.filename)
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
    from tkinter import messagebox
    messagebox.showinfo("Features Extracetd","nobs :- "+str(nobs)+"\nminmax :- "+str(minmax)+"\nmean :- "+str(mean)+"\nvariance :- "+str(variance)+"\nskew :- "+str(skew)+"\nkurtosis :- "+str(kurtosis)+"\nmedian :- "+str(median)+"\nmode :- "+str(mode)+"\nstd :- "+str(std)+"\nlow :- "+str(low)+"\npeak :- "+str(peak)+"\nq25 :- "+str(q25)+"\nq75 :- "+str(q75)+"\niqr :- "+str(iqr))


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

    import sklearn
    from sklearn import svm

    C = 1.0
#svc = svm.SVC(kernel='linear',C=C,gamma=2)
#svc.fit(X_train, y_train)

    from sklearn.metrics import fbeta_score
#print (X_test)
    predictions_test = forest.predict(X_test)
    predictions_test
#predictions_test
    #predictions_test

    if(predictions_test):
        print("MALE")
    else:
        print("FEMALE")
        
def browsedetect():
    try:
        rate,data=wavfile.read(master.filename)

    #rate,data=wavfile.read('/home/user/test/b0476.wav')
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

        import sklearn
        from sklearn import svm

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
            var="Male"
        else:
            var="Female"
        from tkinter import messagebox
        messagebox.showinfo("GENDER DETECTED", "the voice is of a "+var)
    except ValueError:
        messagebox.showinfo("sorry","No gender")
    
   # if(predictions_test):
Label (master,text="     ").grid(row=0,column=0)      
Label (master,text="     ").grid(row=0,column=1)      
Label (master,text="     ").grid(row=0,column=2)      
lab=Label (master,text=" GENDER DETECTION",fg="red",justify='center')
lab.config(bg="black")
lab.grid(row=0,column=3)    
lab1=Label (master,text=" RECORDING AND DETECT",fg="black",justify='center')
lab1.grid(row=1,column=3) 
Label (master,text="     ").grid(row=1,column=2)      

Label (master,text="Record your voice      ",fg="black").grid(row=2,column=0)  
     
Label (master,text="     ").grid(row=2,column=2)      

# Label (master,text="Choose an audio file").grid(row=1,column=0)
#Button(master, text="BROWSE", command=browsefun).grid(row=1, column=2)
b=Button(master, text="RECORD", command=callback,bg="white",fg="black")
b.grid(row=2, column=3)    
Label (master,text="     ").grid(row=3,column=0)      

Label (master,text="Find out the features      ",fg="black").grid(row=4,column=0)          
     
Label (master,text="     ").grid(row=4,column=2)      

Button(master, text="FIND", command=find,bg="white",fg="black").grid(row=4, column=3)
Label (master,text="     ").grid(row=5,column=2)     
Label (master,text="MALE/FEMALE?    ",fg="black").grid(row=6,column=0)
Label (master,text="     ").grid(row=6,column=2)     
Button(master, text="DETECT", command=detect,bg="white",fg="black").grid(row=6, column=3)

Label (master,text="     ").grid(row=7,column=0)      
Label (master,text="     ").grid(row=8,column=1)      
Label (master,text="     ").grid(row=9,column=2)


Label (master,text="     ").grid(row=10,column=0)      
Label (master,text="     ").grid(row=10,column=1)      
Label (master,text="     ").grid(row=10,column=2)      
Label (master,text=" BROWSE AND DETECT   ",fg="black").grid(row=10,column=3)      
    
Label (master,text="     ").grid(row=11,column=2)      

Label (master,text="Browse a file      ",fg="black").grid(row=12,column=0)  
     
Label (master,text="     ").grid(row=12,column=2)      

# Label (master,text="Choose an audio file").grid(row=1,column=0)
#Button(master, text="BROWSE", command=browsefun).grid(row=1, column=2)
Button(master, text="BROWSE", command=browsefun,bg="white",fg="black").grid(row=12, column=3)
     
Label (master,text="     ").grid(row=13,column=0)      

Label (master,text="Find out the features      ",fg="black").grid(row=14,column=0)          
     
Label (master,text="     ").grid(row=14,column=2)      

Button(master, text="FIND", command=browsefind,bg="white",fg="black").grid(row=14, column=3)
Label (master,text="     ").grid(row=15,column=2)     
Label (master,text="MALE/FEMALE?    ",fg="black").grid(row=16,column=0)
Label (master,text="     ").grid(row=16,column=2)     
Button(master, text="DETECT", command=browsedetect,bg="white",fg="black").grid(row=16, column=3)


Label (master,text=" ").grid(row=35,column=0) 
Label (master,text="  ").grid(row=36,column=0) 


Label (master,text="     ").grid(row=38,column=1)
Label (master,text="     ").grid(row=38,column=2) 
Label (master,text="     ").grid(row=38,column=3) 
Label (master,text="     ").grid(row=38,column=4) 
Label (master,text="     ").grid(row=38,column=5) 
Label (master,text="     ").grid(row=38,column=6) 


Button(master, text="EXIT ", command=master.destroy,bg="white",fg="black").grid(row=38, column=9)

master.title("VOICE DETECTION") 
master.mainloop()    
     
    
