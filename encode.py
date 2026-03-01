import numpy as np
from scipy.io.wavfile import write
import sounddevice as sd
# Parametreler
fs = 44100  # Daha standart bir kalite için 8000'den 44100'e çıkardık
duration = 0.04

alphabet = list("ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ ")

low_freqs = np.linspace(500, 1500, 6)
high_freqs = np.linspace(2000, 3500, 5)

# Frekans eşlemesi
freq_map = {}
index = 0
for low in low_freqs:
    for high in high_freqs:
        if index < len(alphabet):
            freq_map[alphabet[index]] = (low, high)
            index += 1

# Kullanıcıdan metin al
text = input("Sese çevrilecek metni girin: ").upper()

signal_list = [] # Hız için liste kullanıp sonra birleştirmek daha iyidir
t = np.linspace(0, duration, int(fs*duration), endpoint=False)

for char in text:
    if char in freq_map:
        f1, f2 = freq_map[char]
        # Genliği 0.5 ile çarparak toplamın 1.0'ı geçmemesini sağladık (Normalizasyon)
        tone = 0.5 * (np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t))
        signal_list.append(tone)
    elif char == " ":
        # Boşluk karakteri için sessizlik ekle
        signal_list.append(np.zeros(int(fs*duration)))

# Tüm parçaları birleştir
if signal_list:
    final_signal = np.concatenate(signal_list)
    # WAV dosyası oluştur
    write("encoded.wav", fs, final_signal.astype(np.float32))
    print("encoded.wav dosyası başarıyla oluşturuldu.")
else:
    print("Geçerli bir metin girilmedi.")
    
sd.play(final_signal, fs)
sd.wait()