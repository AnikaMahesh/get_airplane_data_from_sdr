from rtlsdr import RtlSdr
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
sdr = RtlSdr()
sdr.sample_rate = 2.048e6 # Hz
sdr.center_freq = 139e6   # Hz
sdr.freq_correction = 60  # PPM
sdr.gain = 49.6

fft_size = 512
num_rows = 500
x = sdr.read_samples(4096) # get rid of initial empty samples
print("removed initial samples")

plt.plot(x.real)
plt.plot(x.imag)
plt.legend(["I", "Q"])
plt.savefig("/mnt/c/Users/amahesh/Olin/Projects/SDR/images/rtlsdr-gain2.png", bbox_inches='tight')
plt.close()
x = sdr.read_samples(fft_size*num_rows) # get all the samples we need for the spectrogram
print(len(x))
spectrogram = np.zeros((num_rows, fft_size))
print(spectrogram.shape)
for i in range(num_rows):
    print("row", i)
    if i == 1:
        plt.plot(np.abs(np.fft.fftshift(np.fft.fft(x[i*fft_size:(i+1)*fft_size]))))
        plt.savefig("/mnt/c/Users/amahesh/Olin/Projects/SDR/images/fft.png", bbox_inches='tight')
        plt.close()
    spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[i*fft_size:(i+1)*fft_size])))**2)
extent = [(sdr.center_freq + sdr.sample_rate/-2)/1e6,
            (sdr.center_freq + sdr.sample_rate/2)/1e6,
            len(x)/sdr.sample_rate, 0]
plt.imshow(spectrogram, aspect='auto', extent=extent)
plt.xlabel("Frequency [MHz]")
plt.ylabel("Time [s]")
plt.savefig("/mnt/c/Users/amahesh/Olin/Projects/SDR/images/spectograph_antemma4.png", bbox_inches='tight')
plt.close()
sdr.close()

# reconstructing signals from spectrograms
