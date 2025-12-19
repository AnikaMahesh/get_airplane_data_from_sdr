from rtlsdr import RtlSdr
import numpy as np
from scipy.signal import firwin, lfilter, butter
from scipy.io.wavfile import write
import numpy as np
import matplotlib

matplotlib.use('Agg') 
import matplotlib.pyplot as plt
sdr = RtlSdr()
sdr.sample_rate = 2.048e6 # Hz
sdr.center_freq = 112.7e6  # Hz
sdr.freq_correction = 60  # PPM
sdr.gain = 49.6

fft_size = 512
num_rows = 500
x = sdr.read_samples(4096) # get rid of initial empty samples
print("removed initial samples")

def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)




plt.plot(x.real)
plt.plot(x.imag)
plt.legend(["I", "Q"])
plt.savefig("rtlsdr-gain2.png", bbox_inches='tight')
plt.close()
x = sdr.read_samples(fft_size*num_rows) # get all the samples we need for the spectrogram
print(len(x))
spectrogram = np.zeros((num_rows, fft_size))
print(spectrogram.shape)
for i in range(num_rows):
    print("row", i)
    if i == 1:
        x_fft = x[i*fft_size:(i+1)*fft_size]

        # Window
        window = np.hanning(len(x_fft))
        xw = x_fft * window

        # FFT
        X = np.fft.fftshift(np.fft.fft(xw))
        mag_db = 20 * np.log10(np.abs(X) + 1e-12)

        # Frequency axis (RF, MHz)
        freqs = np.fft.fftshift(
            np.fft.fftfreq(len(X), d=1/sdr.sample_rate)
        )


        freqs_mhz = (freqs + sdr.center_freq) / 1e6
        max_mag = np.max(mag_db)
        max_freq = freqs_mhz[np.argmax(mag_db)]
        # Plot

        plt.figure(figsize=(10,4))
        plt.plot(freqs_mhz, mag_db)
        annot_max(freqs_mhz,mag_db)
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Magnitude (dB)")
        plt.title("FFT Snapshot")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("fft.png")
        plt.annotate('Styled Annotation', xy=(max_freq, max_mag), xytext=(max_freq+0.1, max_mag+0.1),
        bbox=dict(boxstyle="round", fc="lightblue"),
        arrowprops=dict(arrowstyle="fancy", color="red"))
        plt.close()

    spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[i*fft_size:(i+1)*fft_size])))**2)
extent = [(sdr.center_freq + sdr.sample_rate/-2)/1e6,
            (sdr.center_freq + sdr.sample_rate/2)/1e6,
            len(x)/sdr.sample_rate, 0]
plt.imshow(spectrogram, aspect='auto', extent=extent)
plt.xlabel("Frequency [MHz]")
plt.ylabel("Time [s]")
plt.savefig("spectograph_antemma4.png", bbox_inches='tight')
plt.close()



# reconstructing signals from spectrogram

f_target = 112.700e6           # <-- change to the AM channel you want
duration_s = 1.0               # seconds to record

N = int(duration_s * sdr.sample_rate)

# flush a bit
x = sdr.read_samples(N)
sdr.close()

fs = sdr.sample_rate
t = np.arange(len(x)) / fs

# Digital tune (shift target to 0 Hz)
f_offset = f_target - sdr.center_freq
x = x * np.exp(-1j * 2*np.pi * f_offset * t)

# Channel filter (airband is narrow; 25 kHz spacing, voice is much narrower)
chan_bw = 12_000   # Hz (you can try 8k-15k)
numtaps = 129
lp = firwin(numtaps, chan_bw/(fs/2))
x = lfilter(lp, 1.0, x)

# Decimate to audio-friendly rate
decim = int(fs // 48_000)      # aim ~48 kHz
x = x[::decim]
fs2 = fs / decim

# AM demod (envelope)
env = np.abs(x)

# Remove DC + optional AGC-ish normalization
audio = env - np.mean(env)
audio = audio / (np.max(np.abs(audio)) + 1e-12)

# Voice low-pass (~3 kHz)
b, a = butter(4, 3000/(fs2/2))
audio = lfilter(b, a, audio)

# Save WAV (16-bit PCM)
audio_i16 = np.int16(np.clip(audio, -1, 1) * 32767)
write("airband.wav", int(fs2), audio_i16)

print("Wrote airband.wav at", int(fs2), "Hz")

sdr.close()