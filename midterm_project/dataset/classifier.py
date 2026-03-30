import numpy as np
import pickle
from collections import Counter

# F0 sonuçlarını yükle
with open("results_f0.pkl", "rb") as f:
    results = pickle.load(f)
print(f"Yüklendi: {len(results)} dosya")

# Kural tabanlı sınıflandırıcı
def classify(f0):
    if f0 is None:
        return "unknown"
    if f0 < 200:
        return "male"
    elif f0 < 290:
        return "female"
    else:
        return "child"

# Tahminleri hesapla
correct = 0
total = 0
errors = []

confusion = {
    'male':   Counter(),
    'female': Counter(),
    'child':  Counter()
}

for r in results:
    if r['f0'] is None:
        continue
    actual    = r['actual_class']
    predicted = classify(r['f0'])
    confusion[actual][predicted] += 1
    if actual == predicted:
        correct += 1
    else:
        errors.append({
            'file'     : r['file_name'],
            'actual'   : actual,
            'predicted': predicted,
            'f0'       : round(r['f0'], 1)
        })
    total += 1

# Genel doğruluk
accuracy = correct / total * 100
print(f"\nGenel Doğruluk: %{accuracy:.1f}  ({correct}/{total})")

# Sınıf bazlı başarı
print("\nSınıf Bazlı Başarı:")
for sinif in ['male', 'female', 'child']:
    n = sum(confusion[sinif].values())
    acc = confusion[sinif][sinif] / n * 100 if n > 0 else 0
    print(f"  {sinif:8s} → %{acc:.1f}  ({confusion[sinif][sinif]}/{n})")

# Confusion Matrix
print("\nConfusion Matrix:")
print(f"{'':10s} {'male':>8s} {'female':>8s} {'child':>8s}")
for sinif in ['male', 'female', 'child']:
    row = f"{sinif:10s}"
    for pred in ['male', 'female', 'child']:
        row += f"{confusion[sinif][pred]:>8d}"
    print(row)

# İlk 10 hatalı tahmin
print(f"\nToplam hata: {len(errors)}")
print("İlk 10 hatalı tahmin:")
for e in errors[:10]:
    print(f"  {e['file']:40s} Gerçek: {e['actual']:8s} Tahmin: {e['predicted']:8s} F0: {e['f0']} Hz")

# Kaydet
with open("results_classified.pkl", "wb") as f:
    pickle.dump(results, f)
print("\nresults_classified.pkl kaydedildi!")
