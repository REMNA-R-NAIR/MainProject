#reading an audio file

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
    #noise is seen between 50Hz and 60Hz
    filtered_frequencies = [f for f in window_frequencies if 20<f<300 and not 46<f<66] 
print(filtered_frequencies)
#finding spectral features
nobs, minmax, mean, variance, skew, kurtosis = stats.describe(filtered_frequencies)
median    = np.median(filtered_frequencies)
mode      = stats.mode(filtered_frequencies).mode[0]
std       = np.std(filtered_frequencies)
low,peak  = minmax
q75,q25   = np.percentile(freqs, [75 ,25])
iqr       = q75 - q25

print(nobs)
print(minmax)
print(mean)
print(variance)
print(skew) 
print(kurtosis)
print(median)
print(mode)
print(std)
print(low)
print(peak)
print(q75)
print(q25)
print(iqr)

