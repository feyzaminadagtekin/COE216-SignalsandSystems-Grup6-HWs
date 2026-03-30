import numpy as np
import pickle
import pandas as pd

# Sonuçları yükle
with open("results_classified.pkl", "rb") as f:
    results = pickle.load(f)

# Sınıflandırıcı (aynı eşikler)
def classify(f0):
    if f0 is None:
        return "unknown"
    if f0 < 200:
        return "male"
    elif f0 < 290:
        return "female"
    else:
        return "child"

# Her sınıf için istatistik hesapla
siniflar = ['male', 'female', 'child']
tablo = []

for sinif in siniflar:
    f0_list = [r['f0'] for r in results if r['actual_class'] == sinif and r['f0'] is not None]
    dogru   = sum(1 for r in results if r['actual_class'] == sinif and classify(r['f0']) == sinif)
    toplam  = len(f0_list)
    basari  = dogru / toplam * 100 if toplam > 0 else 0

    tablo.append({
        'Sınıf'         : sinif.capitalize(),
        'Örnek Sayısı'  : toplam,
        'Ort. F0 (Hz)'  : round(np.mean(f0_list), 1),
        'Std Sapma (Hz)': round(np.std(f0_list), 1),
        'Başarı (%)'    : round(basari, 1)
    })

df_tablo = pd.DataFrame(tablo)
print(df_tablo.to_string(index=False))

# Excel'e kaydet
df_tablo.to_excel("istatistik_tablosu.xlsx", index=False)
print("\nistatistik_tablosu.xlsx kaydedildi!")