import numpy as np
from scipy.io.wavfile import read
from scipy.fft import fft
import matplotlib.pyplot as plt

fs = 8000
duration = 0.04

alphabet = list("ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ ")

low_freqs = np.linspace(500, 1500, 6)
high_freqs = np.linspace(2000, 3500, 5)

freq_map = {}
index = 0
for low in low_freqs:
    for high in high_freqs:
        if index < len(alphabet):
            freq_map[alphabet[index]] = (low, high)
            index += 1

# ---------------- DECODE ----------------
def decode_wav(filename):
    try:
        fs_read, signal = read(filename)
    except FileNotFoundError:
        print("Hata: Dosya bulunamadı!")
        return
        
    # Genlik eşiğinin doğru çalışması için sinyali normalize ediyoruz
    if signal.dtype == np.int16:
        signal = signal / 32768.0
    else:
        signal = signal / np.max(np.abs(signal))

    samples_per_char = int(fs_read * duration)
    decoded_text = ""
    last_char = ""

    # Grafik için ilk geçerli karakterin verilerini saklayacağız
    first_valid_segment = None
    first_valid_spectrum = None
    first_valid_freqs = None
    first_valid_char = None
    first_valid_f1 = None
    first_valid_f2 = None

    for i in range(0, len(signal), samples_per_char):
        segment = signal[i:i+samples_per_char]
        if len(segment) < samples_per_char:
            continue
            
        # Grafikte zaman düzlemini pencerelemeden önceki saf haliyle göstermek için kopyalıyoruz
        raw_segment = segment.copy()

        window = np.hamming(len(segment))
        segment = segment * window

        spectrum = np.abs(fft(segment))
        freqs = np.fft.fftfreq(len(segment), 1/fs_read)

        half = len(freqs)//2
        spectrum = spectrum[:half]
        freqs = freqs[:half]

        # 1. ÇÖZÜM: Sessizlik (Gürültü) eşiği. Sinyal çok zayıfsa bu döngüyü atla.
        if np.max(spectrum) < 0.5: 
            last_char = ""
            continue

        # 2. ÇÖZÜM: Sinyali Düşük ve Yüksek bantlara bölerek en yüksek 2 tepeyi buluyoruz.
        low_band = np.where((freqs >= 400) & (freqs <= 1600))
        high_band = np.where((freqs >= 1900) & (freqs <= 3600))
        
        if len(low_band[0]) == 0 or len(high_band[0]) == 0:
            continue

        f1_detected = freqs[low_band][np.argmax(spectrum[low_band])]
        f2_detected = freqs[high_band][np.argmax(spectrum[high_band])]

        best_char = None
        min_error = float("inf")

        # 3. ÇÖZÜM: Artık f1_detected kesinlikle düşük, f2_detected yüksek frekans.
        for char, (f1, f2) in freq_map.items():
            error = abs(f1 - f1_detected) + abs(f2 - f2_detected)
            if error < min_error:
                min_error = error
                best_char = char

        # Aynı harfin tekrar tekrar okunmasını engelle (Debouncing)
        if best_char and best_char != last_char:
            decoded_text += best_char
            last_char = best_char
            
            # Grafik çizdirmek için sadece İLK yakalanan harfin verilerini kaydet
            if first_valid_segment is None:
                first_valid_segment = raw_segment
                first_valid_spectrum = spectrum
                first_valid_freqs = freqs
                first_valid_char = best_char
                first_valid_f1 = f1_detected
                first_valid_f2 = f2_detected

    print("\nDecoded Text:", decoded_text)
    
    # -------- GRAFİK ÇİZDİRME BÖLÜMÜ --------
    if first_valid_segment is not None:
        plt.figure(figsize=(10, 8))
        
        # Grafik 1: Tespit edilen ilk karakterin zaman düzlemi
        plt.subplot(2, 1, 1)
        t_segment = np.linspace(0, duration, len(first_valid_segment), endpoint=False)
        plt.plot(t_segment, first_valid_segment, color='blue')
        plt.title(f"Zaman Düzlemi - İlk Tespit Edilen Karakter: '{first_valid_char}' (Süre: 40ms)")
        plt.xlabel("Zaman (s)")
        plt.ylabel("Genlik")
        plt.grid(True, alpha=0.3)

        # Grafik 2: Tespit edilen ilk karakterin Frekans Spektrumu (FFT)
        plt.subplot(2, 1, 2)
        plt.plot(first_valid_freqs, first_valid_spectrum, color='red')
        
        # Bulunan tepe noktalarını grafikte işaretle
        plt.axvline(x=first_valid_f1, color='black', linestyle='--', alpha=0.5, label=f'Low Peak: {first_valid_f1:.1f} Hz')
        plt.axvline(x=first_valid_f2, color='black', linestyle='--', alpha=0.5, label=f'High Peak: {first_valid_f2:.1f} Hz')
        
        plt.title(f"Frekans Spektrumu (FFT) - Karakter: '{first_valid_char}'")
        plt.xlabel("Frekans (Hz)")
        plt.ylabel("Genlik")
        plt.xlim(0, 4000)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    else:
        print("\nGrafik çizdirilecek geçerli bir sinyal bulunamadı.")

# ---------------- MENU ----------------
print(" Sesi metne çevir (Decode)")
filename = input("Wav dosya adını girin (örn: encoded.wav): ")
decode_wav(filename)