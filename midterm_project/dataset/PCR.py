import pandas as pd
import numpy as np
import librosa
import os
import pickle
# 1. Excel oku
df = pd.read_excel("master_metadata.xlsx")
df = df[df['audio_file_present'] == True].reset_index(drop=True)

# 2. Doğru yolu bulan fonksiyon
def find_actual_path(relative_path):
    if os.path.exists(relative_path):
        return relative_path
    parts = relative_path.replace("\\", "/").split("/")
    group_folder = parts[-2]
    file_name    = parts[-1]
    candidates = [
        group_folder,
        group_folder.replace("GROUP_", "GRUP_"),
        group_folder.replace("GRUP_", "GROUP_"),
        group_folder.capitalize(),
    ]
    for candidate in candidates:
        path = os.path.join(candidate, file_name)
        if os.path.exists(path):
            return path
    return None

df['real_path'] = df['audio_relative_path'].apply(find_actual_path)
print(f"Bulunan: {df['real_path'].notna().sum()}")

# 3. Analiz fonksiyonu
def analyze_file(file_path, sr=22050, window_ms=25, hop_ms=10):
    audio, sr = librosa.load(file_path, sr=sr)
    window_size = int(sr * window_ms / 1000)
    hop_size    = int(sr * hop_ms / 1000)
    frames = librosa.util.frame(audio, frame_length=window_size, hop_length=hop_size)
    ste = np.sum(frames ** 2, axis=0)
    zcr = librosa.feature.zero_crossing_rate(
        audio, frame_length=window_size, hop_length=hop_size
    )[0]
    min_len = min(len(ste), len(zcr))
    ste = ste[:min_len]
    zcr = zcr[:min_len]
    frames = frames[:, :min_len]
    voiced_mask = (ste > np.mean(ste) * 0.5) & (zcr < np.mean(zcr) * 1.5)
    return frames, ste, zcr, voiced_mask, sr, hop_size

# 4. Döngü
# 4. Döngü
df_found = df[df['real_path'].notna()].reset_index(drop=True)
results = []
for i, row in df_found.iterrows():
    try:
        frames, ste, zcr, voiced_mask, sr, hop_size = analyze_file(row['real_path'])
        results.append({
            'file_name'   : row['file_name'],
            'actual_class': row['actual_class'],
            'voiced_count': int(np.sum(voiced_mask)),
            'total_frames': len(ste),
            'frames'      : frames,
            'voiced_mask' : voiced_mask,
            'sr'          : sr,
            'hop_size'    : hop_size
        })
    except Exception as e:
        print(f"HATA - {row['file_name']}: {e}")
        continue  # hatalı dosyayı atla, devam et

    if i % 50 == 0:
        print(f"{i}/{len(df_found)} işlendi...")

print(f"Tamamlandı! Toplam: {len(results)} dosya")

from collections import Counter

sinif_dagilim = Counter(r['actual_class'] for r in results)
print("Sınıf dağılımı:")
for sinif, sayi in sinif_dagilim.items():
    print(f"  {sinif}: {sayi} dosya")

oranlar = [r['voiced_count'] / r['total_frames'] * 100 for r in results]
print(f"\nSesli bölge oranı:")
print(f"  Ortalama : %{np.mean(oranlar):.1f}")
print(f"  Min      : %{np.min(oranlar):.1f}")
print(f"  Max      : %{np.max(oranlar):.1f}")

sorunlu = [r for r in results if r['voiced_count'] / r['total_frames'] < 0.1]
print(f"\nSesli bölge %10'un altında olan dosya: {len(sorunlu)}")

with open("results.pkl", "wb") as f:
    pickle.dump(results, f)
print("results.pkl kaydedildi!")