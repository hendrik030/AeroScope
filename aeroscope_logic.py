# aeroscope_logic.py
import datetime, statistics, math
from fitparse import FitFile
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def parse_fit(file_path):
    """Liest FIT-Datei ein und gibt Datenpunkte zurück."""
    fit = FitFile(file_path)
    # Beispiel: sammle Zeitstempel und Herzfrequenz
    times, hr = [], []
    for rec in fit.get_messages("record"):
        data = {f.name: f.value for f in rec}
        if "timestamp" in data and "heart_rate" in data:
            times.append(data["timestamp"])
            hr.append(data["heart_rate"])
    return times, hr

def plot_to_base64(times, values):
    """Erzeugt ein Matplotlib-Diagramm und returniert es als base64-PNG."""
    fig, ax = plt.subplots()
    ax.plot(times, values)
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Herzfrequenz")
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("ascii")
    return img_b64