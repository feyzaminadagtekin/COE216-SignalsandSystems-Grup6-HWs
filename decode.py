import numpy as np
from scipy.io.wavfile import write, read
from scipy.fft import fft

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

    for i in range(0, len(signal), samples_per_char):
        segment = signal[i:i+samples_per_char]
        if len(segment) < samples_per_char:
            continue

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
        # Bu sayede yan yana duran iki frekansı yanlışlıkla seçme hatası ortadan kalkıyor.
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

        # (İsteğe bağlı) Aynı harfin tekrar tekrar okunmasını engelle (Debouncing)
        if best_char and best_char != last_char:
            decoded_text += best_char
            last_char = best_char

    print("\nDecoded Text:", decoded_text)

# ---------------- MENU ----------------
print(" Sesi metne çevir (Decode)")
filename = input("Wav dosya adını girin: ")
decode_wav(filename)