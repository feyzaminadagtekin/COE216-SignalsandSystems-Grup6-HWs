import tkinter as tk
from tkinter import ttk
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# DTMF Standart Frekans Tablosu [cite: 48, 49, 60]
DTMF_TABLE = {
    '1': (697, 1209), '2': (697, 1336), '3': (697, 1477), 'A': (697, 1633),
    '4': (770, 1209), '5': (770, 1336), '6': (770, 1477), 'B': (770, 1633),
    '7': (852, 1209), '8': (852, 1336), '9': (852, 1477), 'C': (852, 1633),
    '*': (941, 1209), '0': (941, 1336), '#': (941, 1477), 'D': (941, 1633),
}

class DTMFApp:
    def __init__(self, root):
        self.root = root
        self.root.title("COE216 - DTMF Arayüzü & Spektrum Analizi")
        self.root.geometry("700x900") # Pencere sığma sorunu için boyut sabitlendi
        
        self.fs = 8000 # Standart örnekleme hızı [cite: 68]
        self.duration = 0.4 # Tuş basım süresi [cite: 69]
        self.setup_ui()

    def setup_ui(self):
        # Grafik Alanı (Üst Kısım) [cite: 55, 85]
        self.fig, (self.ax_t, self.ax_f) = plt.subplots(2, 1, figsize=(6, 7))
        plt.subplots_adjust(hspace=0.4)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tuş Takımı (Alt Kısım) [cite: 53]
        style = ttk.Style()
        style.configure('TButton', font=('Arial', 11, 'bold'))
        
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(side=tk.BOTTOM, pady=30)
        
        keys = [['1', '2', '3', 'A'], ['4', '5', '6', 'B'], ['7', '8', '9', 'C'], ['*', '0', '#', 'D']]
        
        for r, row in enumerate(keys):
            for c, key in enumerate(row):
                btn = ttk.Button(btn_frame, text=key, width=8, command=lambda k=key: self.play(k))
                btn.grid(row=r, column=c, padx=5, pady=5)

    def play(self, key):
        fl, fh = DTMF_TABLE[key]
        t = np.linspace(0, self.duration, int(self.fs * self.duration), endpoint=False)
        
        # Sinyal Sentezi ve Normalizasyon (Clipping önlemek için 0.5 ile çarpıldı) [cite: 48, 73, 75]
        sig = (np.sin(2 * np.pi * fl * t) + np.sin(2 * np.pi * fh * t)) * 0.5
        sd.play(sig, self.fs) # Sesi çal [cite: 76]
        
        # Zaman Grafiği Güncelleme [cite: 55]
        self.ax_t.clear()
        self.ax_t.plot(t[:250], sig[:250], 'b') # Net görünüm için bir kesit
        self.ax_t.set_title(f"Zaman Düzlemi (Tuş: {key})", fontsize=10, fontweight='bold')
        self.ax_t.grid(True, alpha=0.3)
        
        # Spektrum (FFT) Güncelleme - Artı Puan [cite: 85]
        self.ax_f.clear()
        f = np.fft.fftfreq(len(sig), 1/self.fs)
        mag = np.abs(np.fft.fft(sig))
        mask = (f >= 0) & (f <= 2000)
        self.ax_f.plot(f[mask], mag[mask], 'r')
        self.ax_f.set_title(f"Frekans Spektrumu ({fl}Hz & {fh}Hz)", fontsize=10, fontweight='bold')
        self.ax_f.grid(True, alpha=0.3)
        
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = DTMFApp(root)
    root.mainloop()