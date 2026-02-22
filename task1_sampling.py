import numpy as np
import matplotlib.pyplot as plt

# 1. Parametre Belirleme (08 + 53 + 19 = 80 Hz) 
f0 = 80  
f1 = f0
f2 = f0 / 2
f3 = 10 * f0

# 2. Teknik Ayarlar (Nyquist: fs > 2*f3) 
fs = 8000 # Daha pürüzsüz görünüm ve ses standartı için 8000 Hz seçildi 
t = np.arange(0, 3/f2, 1/fs) # En düşük frekansın en az 3 periyodunu gösterir 

# 3. Sinyal Üretimi 
s1 = np.sin(2 * np.pi * f1 * t)
s2 = np.sin(2 * np.pi * f2 * t)
s3 = np.sin(2 * np.pi * f3 * t)
s_sum = s1 + s2 + s3 # Üç sinyalin toplamı 

# 4. Görselleştirme (4 Subplot) 
plt.figure(figsize=(12, 10)) # Pencere boyutu 

signals = [
    (s1, f'Sinyal 1 (f1 = {f1} Hz)', 'b'),
    (s2, f'Sinyal 2 (f2 = {f2} Hz)', 'g'),
    (s3, f'Sinyal 3 (f3 = {f3} Hz)', 'r'),
    (s_sum, 'Üç Sinyalin Toplamı (s1 + s2 + s3)', 'm')
]

for i, (sig, title, color) in enumerate(signals, 1):
    plt.subplot(4, 1, i)
    plt.plot(t, sig, color)
    plt.title(title, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    if i == 4: plt.xlabel('Zaman (s)')

plt.subplots_adjust(hspace=0.6) # Grafikler arası boşluk artırıldı
plt.tight_layout()
plt.show()