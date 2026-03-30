import numpy as np
import matplotlib.pyplot as plt
import librosa

# Herhangi bir ses dosyası seç (örnek olarak ilk male dosyası)
ornek_dosya = r"C:\Users\Elif\Desktop\dataset\GROUP_06\G06_D03_M_20_Shocked_C5.wav"  # kendi dosya yolunla değiştir

# Dosyayı yükle
audio, sr = librosa.load(ornek_dosya, sr=22050)

# Tek bir pencere al (ortadan)
window_size = int(sr * 0.025)  # 25ms
mid = len(audio) // 2
frame = audio[mid:mid + window_size]

# --- OTOKORELASYON ---
corr = np.correlate(frame, frame, mode='full')
corr = corr[len(corr)//2:]
corr = corr / corr[0]  # normalize et

lag_min = int(sr / 500)
lag_max = int(sr / 50)
peak_lag = np.argmax(corr[lag_min:lag_max]) + lag_min
f0_autocorr = sr / peak_lag

# --- FFT ---
N = len(frame)
fft_magnitude = np.abs(np.fft.rfft(frame))
freqs = np.fft.rfftfreq(N, d=1/sr)

# Sadece 50-500 Hz aralığına bak
fft_range = (freqs >= 50) & (freqs <= 500)
f0_fft = freqs[fft_range][np.argmax(fft_magnitude[fft_range])]

# --- GRAFİK ---
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Otokorelasyon grafiği
lags = np.arange(len(corr[:lag_max+50]))
axes[0].plot(sr / np.maximum(lags[lag_min:], 1), corr[lag_min:lag_max+50], color='steelblue')
axes[0].axvline(f0_autocorr, color='red', linestyle='--', label=f'F0 = {f0_autocorr:.1f} Hz')
axes[0].set_title('Otokorelasyon Yöntemi')
axes[0].set_xlabel('Frekans (Hz)')
axes[0].set_ylabel('Otokorelasyon')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# FFT grafiği
axes[1].plot(freqs[fft_range], fft_magnitude[fft_range], color='darkorange')
axes[1].axvline(f0_fft, color='red', linestyle='--', label=f'F0 = {f0_fft:.1f} Hz')
axes[1].set_title('FFT Yöntemi')
axes[1].set_xlabel('Frekans (Hz)')
axes[1].set_ylabel('Genlik')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle(f'Otokorelasyon vs FFT Karşılaştırması\nOtokorelasyon: {f0_autocorr:.1f} Hz  |  FFT: {f0_fft:.1f} Hz', 
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("autocorr_vs_fft.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"Otokorelasyon F0: {f0_autocorr:.1f} Hz")
print(f"FFT F0         : {f0_fft:.1f} Hz")
print("autocorr_vs_fft.png kaydedildi!")