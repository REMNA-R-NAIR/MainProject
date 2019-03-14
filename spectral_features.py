
import wave


def read_wave(filename):

    fp=wave.open(filename,'r') 
    nchannels=fp.getnchannels()
    framerate=fp.getframerate()
    nframes=fp.getnframes()
    sampwidth=fp.getsampwidth()

    z_str=fp.readframes(nframes)
    fp.close()

    dtype_map={1:np.uint8,2:np.uint16}
    ys=np.frombuffer(z_str,dtype=dtype_map[sampwidth])

    waveObject=Wave(ys,framerate=framerate)

    return waveObject


class Wave:

    def __init__(self,ys,ts=None,framerate=None):

        # ys:wave array
        # ts:array of time


        self.ys=np.asanyarray(ys)
        self.framerate=framerate

        if ts is None:
            self.ts =np.arange(len(ys))/self.framerate
        else:
            self.ts=ts

    def make_spectrum(self):

        n=len(self.ys);
        d=1/self.framerate;

        hs = np.fft.rfft(self.ys)
        fs = np.fft.rfftfreq(n, d)
        return Spectrum(hs,fs,self.framerate)
    def spec(self):

       # n=len(self.ys);
        #d=1/self.framerate;

        hs = np.fft.rfft(self.ys)
        return hs 
    def freq(self):

        n=len(self.ys);
        d=1/self.framerate;

        fs = np.fft.rfftfreq(n, d)
        return fs 
    def spectral_properties(self,y,f):
    
        spec = y
        freq = f
        spec = np.abs(spec)
        amp = spec / spec.sum()
        mean = (freq * amp).sum()
        sd = np.sqrt(np.sum(amp * ((freq - mean) ** 2)))
        amp_cumsum = np.cumsum(amp)
        median = freq[len(amp_cumsum[amp_cumsum <= 0.5]) + 1]
        mode = freq[amp.argmax()]
        Q25 = freq[len(amp_cumsum[amp_cumsum <= 0.25]) + 1]
        Q75 = freq[len(amp_cumsum[amp_cumsum <= 0.75]) + 1]
        IQR = Q75 - Q25
        z = amp - amp.mean()
        w = amp.std()
        skew = ((z ** 3).sum() / (len(spec) - 1)) / w ** 3
        kurt = ((z ** 4).sum() / (len(spec) - 1)) / w ** 4
        
        
        print("mean:",mean)
        print("sd:",sd)
        print("median:",median)
        print("mode:",mode)
        print("Q25:",Q25)
        print("Q75:",Q75)
        print("IQR:",IQR)
        print("skew:",skew)
        print("kurt:",kurt)


    


class Spectrum:

    def __init__(self,hs,fs,framerate):

        # hs : array of amplitudes (real or complex)
        # fs : array of frequencies

        self.hs=np.asanyarray(hs)
        self.fs=np.asanyarray(fs)
        self.framerate=framerate

    @property
    def amps(self):
        return np.absolute(self.hs)

    def plot(self, high=None):

       plt.plot(self.fs, self.amps)

data=read_wave('/home/user/project/f.wav') 
spectrum=data.make_spectrum()
spec=data.spec()
fre=data.freq()
data.spectral_properties(spec,fre)
spectrum.plot()


