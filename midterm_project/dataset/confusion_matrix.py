import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter, defaultdict

# Sonuçları yükle
with open("results_final.pkl", "rb") as f:
    results = pickle.load(f)

# Excel'den duygu bilgilerini al
df_meta = pd.read_excel("master_metadata.xlsx")

# file_name -> feeling eşlemesi
emotion_map = dict(zip(df_meta["file_name"], df_meta["feeling"]))

def classify(f0):
    if f0 is None:
        return None
    if f0 < 200:
        return "male"
    elif f0 < 290:
        return "female"
    else:
        return "child"

siniflar = ['male', 'female', 'child']
sinif_tr = ['Erkek', 'Kadın', 'Çocuk']

# 🔥 GERÇEK CONFUSION MATRIX
matrix = np.zeros((3, 3), dtype=int)

# duygu sayacı
emotion_breakdown = defaultdict(Counter)

for r in results:
    if r['f0'] is None:
        continue

    actual = r['actual_class']
    predicted = classify(r['f0'])

    i = siniflar.index(actual)
    j = siniflar.index(predicted)

    matrix[i][j] += 1

    # sadece hatalar için duygu tut
    if actual != predicted:
        fname = r.get("file_name")
        emotion = emotion_map.get(fname, None)

        if pd.notna(emotion):
            emotion_breakdown[(actual, predicted)][str(emotion)] += 1

# Grafik
fig, ax = plt.subplots(figsize=(12, 9))
im = ax.imshow(matrix, cmap='Blues')

ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(sinif_tr, fontsize=13)
ax.set_yticklabels(sinif_tr, fontsize=13)
ax.set_xlabel("Tahmin Edilen", fontsize=13, fontweight='bold')
ax.set_ylabel("Gerçek Sınıf", fontsize=13, fontweight='bold')
ax.set_title("Confusion Matrix", fontsize=15, fontweight='bold', pad=15)

# Hücre yazıları
for i in range(3):
    for j in range(3):

        actual = siniflar[i]
        predicted = siniflar[j]
        sayi = matrix[i][j]

        # DOĞRU → sadece sayı
        if i == j:
            text = str(sayi)

        else:
            if sayi == 0:
                text = "0"
            else:
                sayac = emotion_breakdown[(actual, predicted)]

                top2 = sayac.most_common(2)
                top_sum = sum(v for _, v in top2)
                others = sayi - top_sum

                lines = [str(sayi)]
                lines += [f"{duygu}:{adet}" for duygu, adet in top2]

                if others > 0:
                    lines.append(f"diğer:{others}")

                text = "\n".join(lines)

        renk = "white" if sayi > matrix.max() / 2 else "black"

        ax.text(j, i, text,
                ha='center', va='center',
                fontsize=10,
                fontweight='bold',
                color=renk)

plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("confusion_matrix_duygu.png", dpi=150, bbox_inches='tight')
plt.show()
print("confusion_matrix_duygu.png kaydedildi!")