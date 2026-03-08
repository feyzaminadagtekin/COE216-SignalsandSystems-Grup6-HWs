import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import get_window
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# -----------------------------
# PARAMETERS
# -----------------------------
FRAME_SIZE_MS = 20
OVERLAP = 0.5
HANGOVER = 3

# -----------------------------
# SELECT AUDIO FILE
# -----------------------------
Tk().withdraw()
file_path = askopenfilename(title="Select a WAV file", filetypes=[("WAV files","*.wav")])

signal, fs = sf.read(file_path)

# stereo ise mono yap
if len(signal.shape) > 1:
    signal = signal[:,0]

# -----------------------------
# NORMALIZATION
# -----------------------------
signal = signal / np.max(np.abs(signal))

# -----------------------------
# FRAME SETTINGS
# -----------------------------
frame_size = int(fs * FRAME_SIZE_MS / 1000)
hop_size = int(frame_size * (1 - OVERLAP))

num_frames = int((len(signal) - frame_size) / hop_size)

window = get_window("hamming", frame_size)

frames = []
energy = []
zcr = []

# -----------------------------
# FRAME PROCESSING
# -----------------------------
for i in range(num_frames):

    start = i * hop_size
    end = start + frame_size

    frame = signal[start:end]

    if len(frame) < frame_size:
        break

    # Hamming uygulanmış çerçeveyi analiz için saklıyoruz
    windowed_frame = frame * window
    frames.append(windowed_frame)

    # Energy
    e = np.sum(windowed_frame ** 2)
    energy.append(e)

    # Zero Crossing Rate
    zero_cross = np.sum(np.abs(np.diff(np.sign(windowed_frame)))) / 2
    z = zero_cross / len(windowed_frame)
    zcr.append(z)

energy = np.array(energy)
zcr = np.array(zcr)

# -----------------------------
# NOISE THRESHOLD (first 200ms)
# -----------------------------
noise_samples = int(0.2 * fs)
noise_frames = int(noise_samples / hop_size)

noise_energy = np.mean(energy[:noise_frames])

threshold_energy = noise_energy * 3

# -----------------------------
# VAD DECISION
# -----------------------------
vad = energy > threshold_energy

# Hangover smoothing (Zincirleme reaksiyonu önlemek için kopya kullanıyoruz)
vad_smoothed = vad.copy()

for i in range(len(vad)):
    if not vad[i]:
        prev = vad[max(0, i-HANGOVER):i]
        
        if np.sum(prev) > 0:
            vad_smoothed[i] = True

vad = vad_smoothed

# -----------------------------
# VOICED / UNVOICED
# -----------------------------
zcr_threshold = np.mean(zcr) * 1.5

voiced = np.zeros(len(vad))
unvoiced = np.zeros(len(vad))

for i in range(len(vad)):
    if vad[i]:
        if energy[i] > threshold_energy and zcr[i] < zcr_threshold:
            voiced[i] = 1
        else:
            unvoiced[i] = 1

# -----------------------------
# RECONSTRUCT SPEECH (CIZIRTI ÇÖZÜMÜ)
# -----------------------------
# Sinyalin orijinal boyutunda bir maske oluşturuyoruz
keep_samples = np.zeros(len(signal), dtype=bool)

for i in range(num_frames):
    if vad[i]:
        start = i * hop_size
        end = start + frame_size
        # Sadece konuşma olan bölgeleri maskeliyoruz
        keep_samples[start:end] = True

# Orijinal sinyalden sadece maskelenmiş (konuşma olan) kısımları alıyoruz
clean_signal = signal[keep_samples]

# Temizlenmiş ve sessizliği atılmış sesi kaydediyoruz
sf.write("clean_speech.wav", clean_signal, fs)

# -----------------------------
# COMPRESSION CALCULATION
# -----------------------------
original_duration = len(signal) / fs
new_duration = len(clean_signal) / fs

compression = ((original_duration - new_duration) / original_duration) * 100

print("\n----- RESULTS -----")
print("Original Duration:", round(original_duration, 2), "seconds")
print("New Duration:", round(new_duration, 2), "seconds")
print("Compression:", round(compression, 2), "%")

# -----------------------------
# TIME AXIS
# -----------------------------
time = np.arange(len(signal)) / fs

# -----------------------------
# PLOTS
# -----------------------------
plt.figure(figsize=(12, 10))

# Original signal
plt.subplot(4, 1, 1)
plt.plot(time, signal, color='gray')
plt.title("Original Audio Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Energy
plt.subplot(4, 1, 2)
plt.plot(energy, color='blue')
plt.axhline(threshold_energy, color='red', linestyle='--', label='Noise Threshold')
plt.title("Short-Time Energy")
plt.legend()

# ZCR
plt.subplot(4, 1, 3)
plt.plot(zcr, color='purple')
plt.axhline(zcr_threshold, color='red', linestyle='--', label='ZCR Threshold')
plt.title("Zero Crossing Rate")
plt.legend()

# VAD + Voiced/Unvoiced
plt.subplot(4, 1, 4)
plt.plot(time, signal, color='lightgray') # Sinyali arka plana silik çiziyoruz

for i in range(len(vad)):
    start_time = (i * hop_size) / fs
    end_time = (start_time + (frame_size / fs))
    
    if voiced[i]:
        plt.axvspan(start_time, end_time, color="green", alpha=0.4)
    elif unvoiced[i]:
        plt.axvspan(start_time, end_time, color="yellow", alpha=0.4)

plt.title("Speech Classification (Green=Voiced, Yellow=Unvoiced)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()