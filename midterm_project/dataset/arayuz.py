import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import librosa
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ============================================================
# GENEL DOĞRULUĞU HESAPLA
# ============================================================

def classify(f0):
    if f0 is None:
        return None
    if f0 < 200:
        return "male"
    elif f0 < 290:
        return "female"
    else:
        return "child"

SINIF_TR = {"male": "Erkek", "female": "Kadın", "child": "Çocuk"}
RENKLER  = {"Erkek": "#4A90D9", "Kadın": "#E91E8C", "Çocuk": "#F5A623"}

with open("results_final.pkl", "rb") as f:
    results = pickle.load(f)

correct  = sum(1 for r in results if r['f0'] is not None and classify(r['f0']) == r['actual_class'])
total    = sum(1 for r in results if r['f0'] is not None)
accuracy = correct / total * 100

# ============================================================
# ANALİZ FONKSİYONU
# ============================================================

def analyze(file_path):
    audio, sr = librosa.load(file_path, sr=22050)
    window_size = int(sr * 0.025)
    hop_size    = int(sr * 0.010)
    frames = librosa.util.frame(audio, frame_length=window_size, hop_length=hop_size)
    ste    = np.sum(frames ** 2, axis=0)
    zcr    = librosa.feature.zero_crossing_rate(audio, frame_length=window_size, hop_length=hop_size)[0]
    min_len = min(len(ste), len(zcr))
    ste, zcr, frames = ste[:min_len], zcr[:min_len], frames[:, :min_len]
    voiced_mask = (ste > np.mean(ste) * 0.5) & (zcr < np.mean(zcr) * 1.5)

    voiced_frames = frames[:, voiced_mask]
    f0_values     = []
    if voiced_frames.shape[1] > 0:
        for i in range(voiced_frames.shape[1]):
            frame    = voiced_frames[:, i]
            corr     = np.correlate(frame, frame, mode='full')
            corr     = corr[len(corr)//2:]
            lag_min  = max(1, int(sr / 500))
            lag_max  = min(int(sr / 50), len(corr) - 1)
            search   = corr[lag_min:lag_max]
            if len(search) == 0:
                continue
            peak_lag = np.argmax(search) + lag_min
            f0_values.append(sr / peak_lag)

    f0_mean = np.mean(f0_values) if f0_values else None
    return ste, zcr, f0_values, voiced_mask, f0_mean, hop_size, sr

# ============================================================
# ARAYÜZ
# ============================================================

root = tk.Tk()
root.title("Ses Sinyali Analizi ve Cinsiyet Sınıflandırması")
root.geometry("1000x750")
root.configure(bg="#1e1e2e")

# Üst bar
top_frame = tk.Frame(root, bg="#2a2a3e", pady=10)
top_frame.pack(fill="x")

btn_sec = tk.Button(top_frame, text="📂 Ses Dosyası Seç", font=("Arial", 11, "bold"),
                    bg="#4A90D9", fg="white", padx=15, pady=6, relief="flat",
                    cursor="hand2")
btn_sec.pack(side="left", padx=15)

lbl_dosya = tk.Label(top_frame, text="Henüz dosya seçilmedi",
                     font=("Arial", 10), bg="#2a2a3e", fg="#aaaacc")
lbl_dosya.pack(side="left", padx=10)

lbl_dogruluk = tk.Label(top_frame, text=f"🎯 Sistem Doğruluğu: %{accuracy:.1f}",
                         font=("Arial", 11, "bold"), bg="#2a2a3e", fg="#00e5a0")
lbl_dogruluk.pack(side="right", padx=20)

# Tahmin kutusu
tahmin_frame = tk.Frame(root, bg="#1e1e2e", pady=10)
tahmin_frame.pack(fill="x")

lbl_tahmin = tk.Label(tahmin_frame, text="—", font=("Arial", 28, "bold"),
                       bg="#1e1e2e", fg="white")
lbl_tahmin.pack()

lbl_f0 = tk.Label(tahmin_frame, text="", font=("Arial", 12),
                   bg="#1e1e2e", fg="#aaaacc")
lbl_f0.pack()

# Grafik alanı
fig, axes = plt.subplots(3, 1, figsize=(10, 6), facecolor="#1e1e2e")
for ax in axes:
    ax.set_facecolor("#2a2a3e")
    ax.tick_params(colors="white")
    ax.title.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=5)

# ============================================================
# DOSYA SEÇ VE ANALİZ ET
# ============================================================

def dosya_sec():
    file_path = filedialog.askopenfilename(filetypes=[("WAV dosyaları", "*.wav")])
    if not file_path:
        return

    lbl_dosya.config(text=os.path.basename(file_path))
    lbl_tahmin.config(text="Analiz ediliyor...", fg="white")
    root.update()

    ste, zcr, f0_values, voiced_mask, f0_mean, hop_size, sr = analyze(file_path)

    tahmin    = classify(f0_mean)
    tahmin_tr = SINIF_TR.get(tahmin, "Bilinmiyor")
    renk      = RENKLER.get(tahmin_tr, "#ffffff")
    f0_label  = f"Ortalama F0: {f0_mean:.1f} Hz" if f0_mean else "F0 hesaplanamadı"

    lbl_tahmin.config(text=f"🧑 {tahmin_tr}", fg=renk)
    lbl_f0.config(text=f0_label)

    # Grafikleri güncelle
    zaman = np.arange(len(zcr)) * hop_size / sr

    for ax in axes:
        ax.clear()
        ax.set_facecolor("#2a2a3e")
        ax.tick_params(colors="white")
        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")

    axes[0].plot(zaman, zcr, color='#ff6b6b')
    axes[0].set_title("ZCR (Sıfır Geçiş Oranı)")
    axes[0].set_xlabel("Zaman (s)")
    axes[0].grid(True, alpha=0.2)

    if f0_values:
        voiced_times = zaman[voiced_mask][:len(f0_values)]
        axes[1].plot(voiced_times, f0_values, color='#6bffb8')
    axes[1].set_title("F0 (Pitch) — Sesli Bölgeler")
    axes[1].set_xlabel("Zaman (s)")
    axes[1].set_ylabel("Hz")
    axes[1].grid(True, alpha=0.2)

    axes[2].plot(zaman, ste, color='#6baeff')
    axes[2].fill_between(zaman, ste, alpha=0.3, color='#6baeff')
    axes[2].set_title("STE (Kısa Süreli Enerji)")
    axes[2].set_xlabel("Zaman (s)")
    axes[2].grid(True, alpha=0.2)

    fig.tight_layout()
    canvas.draw()

btn_sec.config(command=dosya_sec)
root.mainloop()