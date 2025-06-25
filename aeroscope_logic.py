# aeroscope_logic.py

import io
import base64
import datetime
import statistics
import math
from statistics import median
from datetime import timedelta
from zoneinfo import ZoneInfo  # Python 3.9+
import numpy as np
import matplotlib.pyplot as plt
from fitparse import FitFile
import mplcursors
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# 1) HIER FÜGST DU EINMALIG DEINEN ORIGINAL-CODE AUS AeroScope_0.31.py EIN,
#    und zwar **ohne** alle Zeilen, die mit tkinter, FigureCanvasTkAgg oder
#    ImageTk zu tun haben. Einfach das komplette Logik-/Plot-Modul einfügen.
#
#    **Wichtig:** Platziere deinen gesamten Funktions- und Klassen-Code
#    (parse, calc_*-Funktionen, plot_*-Funktionen, Hilfsfunktionen)
#    *unter* dieser Kommentar-Markierung und *oberhalb* der run_analysis.
# ─────────────────────────────────────────────────────────────────────────────

#!/usr/bin/env python3
import datetime
import statistics
import math
from statistics import median
from functools import partial
from datetime import timedelta
from zoneinfo import ZoneInfo  # Erfordert Python 3.9+
import numpy as np
import matplotlib.pyplot as plt
from fitparse import FitFile
import mplcursors

# -----------------------------
# Scrollbarer Frame für die Ereigniszuordnung
# -----------------------------
class ScrollableFrame(ttk.Frame):
    def __init__(self, container, width=1250, height=100, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self, width=width, height=height)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.scrollable_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw", tags="frame")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        canvas_width = event.width
        self.canvas.itemconfig("frame", width=canvas_width)

# -----------------------------
# Hilfsfunktionen
# -----------------------------
def round_dt(dt):
    if dt.microsecond >= 500000:
        dt += datetime.timedelta(seconds=1)
    return dt.replace(microsecond=0)

def convert_to_berlin(dt):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    berlin_tz = ZoneInfo("Europe/Berlin")
    offset = berlin_tz.utcoffset(dt)
    return round_dt(dt + offset)

def interpolate_time(t1, t2, v1, v2, threshold):
    if v1 == v2:
        return t1
    fraction = (v1 - threshold) / (v1 - v2)
    dt_seconds = (t2 - t1).total_seconds()
    return t1 + datetime.timedelta(seconds=dt_seconds * fraction)

def plot_speed_events_manual(records, events, set_a_indices, set_b_indices, set_c_indices, set_d_indices, ax):
    ax.clear()
    set_colors = {'A': 'green', 'B': 'orange', 'C': 'purple', 'D': 'red'}
    times = [t for (t, _) in records]
    speeds = [v * 3.6 for (_, v) in records]  # Umrechnung in km/h
    plotted_lines = []
    event_labels = []

    def plot_event_set(event_indices, color):
        for idx in event_indices:
            start_time, end_time, duration = events[idx - 1]
            event_times = [t for t in times if start_time <= t <= end_time]
            if not event_times:
                continue
            start_idx = times.index(event_times[0])
            end_idx = times.index(event_times[-1])
            event_speeds = speeds[start_idx:end_idx + 1]
            event_times_rel = [(t - start_time).total_seconds() for t in event_times]
            line, = ax.plot(event_times_rel, event_speeds, color=color)
            plotted_lines.append(line)
            event_labels.append(f"Event {idx}")
            # Optional: statische Textanzeige am Anfang und Ende
            ax.text(0, event_speeds[0], str(idx), fontsize=10, color=color, ha='left', va='bottom')
            ax.text(event_times_rel[-1], event_speeds[-1], str(idx), fontsize=10, color=color, ha='right', va='top')

    plot_event_set(set_a_indices, set_colors['A'])
    plot_event_set(set_b_indices, set_colors['B'])
    plot_event_set(set_c_indices, set_colors['C'])
    plot_event_set(set_d_indices, set_colors['D'])

    # Legende
    legend_elements = [plt.Line2D([0], [0], color=clr, lw=4) for clr in set_colors.values()]
    ax.legend(legend_elements, ['Set A', 'Set B', 'Set C', 'Set D'], loc='upper right')
    ax.set_xlabel("Verstrichene Zeit (Sek.)")
    ax.set_ylabel("Geschwindigkeit (km/h)")
    ax.grid(True)

    # Interaktive Cursor (Mouseover)
    cursor = mplcursors.cursor(plotted_lines, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        index = plotted_lines.index(sel.artist)
        sel.annotation.set_text(event_labels[index])
# -----------------------------
# Analyse-Funktion
# -----------------------------
def run_analysis():
    # Eingaben einlesen
    filepath = file_entry_widget.get()
    if not filepath:
        messagebox.showerror("Fehler", "Bitte erst eine FIT-Datei auswählen.")
        return
    try:
        upper_threshold = float(upper_threshold_entry_widget.get())
        lower_threshold = float(lower_threshold_entry_widget.get())
        min_duration = float(min_duration_entry_widget.get())
        max_duration = float(max_duration_entry_widget.get())
        lap_window_val = float(lap_window_entry_widget.get())
        system_mass = float(system_mass_entry_widget.get())
        air_pressure = float(air_pressure_entry_widget.get())
        air_temp = float(air_temp_entry_widget.get())
        crr = float(crr_entry_widget.get())
        slope_percent = float(slope_percent_entry_widget.get())
        filter_window = int(filter_window_entry_widget.get())

    except ValueError:
        messagebox.showerror("Fehler", "Bitte gültige Zahlenwerte eingeben.")
        return

    # FIT-Datei laden
    try:
        fitfile = FitFile(filepath)
    except Exception as e:
        messagebox.showerror("Fehler", f"Fehler beim Öffnen der Datei:\n{e}")
        return

    # 'record'-Daten (Timestamp und speed) einlesen
    records = []
    for record in fitfile.get_messages("record"):
        data = {field.name: field.value for field in record}
        if "speed" in data and data["speed"] is not None:
            records.append((data["timestamp"], data["speed"]))

    # Sortieren nach Zeit
    records.sort(key=lambda x: x[0])
    if len(records) >= filter_window:
        # Entpacken
        timestamps, speeds = zip(*records)
        speeds = np.array(speeds)

        # Glättung
        weights = np.ones(filter_window) / filter_window
        smoothed_speeds = np.convolve(speeds, weights, mode='same')

        # Neue geglättete Liste
        records = list(zip(timestamps, smoothed_speeds))

    # 'lap'-Daten (start_time) einlesen
    lap_times = []
    for lap in fitfile.get_messages("lap"):
        lap_data = {field.name: field.value for field in lap}
        if "start_time" in lap_data:
            lap_times.append(lap_data["start_time"])
    lap_times.sort()

    # Bremsereignisse mittels linearer Interpolation ermitteln
    events = []
    event_start = None
    thresh_up = upper_threshold / 3.6
    thresh_low = lower_threshold / 3.6
    i = 0
    while i < len(records) - 1:
        t1, v1 = records[i]
        t2, v2 = records[i + 1]
        if event_start is None and v1 >= thresh_up and v2 < thresh_up:
            event_start = interpolate_time(t1, t2, v1, v2, thresh_up)
        elif event_start and v1 > thresh_low and v2 <= thresh_low:
            event_end = interpolate_time(t1, t2, v1, v2, thresh_low)
            duration = (event_end - event_start).total_seconds()

            if min_duration <= duration <= max_duration:
                events.append((event_start, event_end, duration))
            event_start = None
        elif event_start and min_duration > (t2 - event_start).total_seconds() > max_duration:
            event_start = None
        i += 1

    if not events:
        messagebox.showinfo("Ergebnis", "Keine passenden Bremsereignisse gefunden.")
        return

    # Standardzuordnung: "Automatisch"
    event_assignments = {i: "auto" for i in range(1, len(events) + 1)}

    # Ergebnisbereich leeren
    for widget in result_frame.winfo_children():
        widget.destroy()

    # --- Ereigniszuordnung in einem scrollbaren Frame und in mehreren Spalten ---
    assignment_label = ttk.Label(result_frame, text="Ereignis Zuordnung:")
    assignment_label.pack(anchor="w")
    scroll_frame = ScrollableFrame(result_frame, width=1250, height=100)
    scroll_frame.pack(fill="x", pady=5)

    options = ["Automatisch", "Set A", "Set B", "Set C", "Set D", "Ausblenden"]

    def on_assignment_change(event_idx, var):
        mapping = {
            "Automatisch": "auto",
            "Set A": "A",
            "Set B": "B",
            "Set C": "C",
            "Set D": "D",
            "Ausblenden": "ignore"
        }
        event_assignments[event_idx] = mapping.get(var.get(), "auto")
        update_display()

    columns = 10
    for i in range(1, len(events) + 1):
        row = (i - 1) // columns
        col = (i - 1) % columns
        frame_event = ttk.Frame(scroll_frame.scrollable_frame)
        frame_event.grid(row=row, column=col, padx=5, pady=2, sticky="w")
        lbl = ttk.Label(frame_event, text=f"Event {i}:")
        lbl.pack(side="left")
        var = tk.StringVar(value="Automatisch")
        om = ttk.Combobox(frame_event, textvariable=var, values=options, width=12, state="readonly")
        om.pack(side="left", padx=2)
        om.bind("<<ComboboxSelected>>", lambda evt, idx=i, v=var: on_assignment_change(idx, v))

    # Funktion zur Aktualisierung des Berichts und Plots
    def update_display():
        # Lösche alle Widgets im Ergebnisbereich außer dem Zuordnungsbereich
        for widget in result_frame.winfo_children():
            if widget not in (assignment_label, scroll_frame):
                widget.destroy()

        event_list_str = "Gefundene Bremsereignisse:\n\n"
        set_a_indices, set_b_indices = [], []
        set_c_indices, set_d_indices = [], []
        unassigned_indices = []

        for idx, (start, end, duration) in enumerate(events, start=1):
            assign = event_assignments.get(idx, "auto")
            if assign == "ignore":
                continue
            elif assign == "auto":
                laps_after = [lt for lt in lap_times if end <= lt <= end + timedelta(seconds=lap_window_val)]
                if len(laps_after) == 1:
                    assign = "A"
                elif len(laps_after) == 2:
                    assign = "B"
                elif len(laps_after) == 3:
                    assign = "C"
                elif len(laps_after) == 4:
                    assign = "D"
                else:
                    assign = "unzugeordnet"
            if assign == "A":
                set_a_indices.append(idx)
            elif assign == "B":
                set_b_indices.append(idx)
            elif assign == "C":
                set_c_indices.append(idx)
            elif assign == "D":
                set_d_indices.append(idx)
            else:
                unassigned_indices.append(idx)

            event_list_str += (f"{idx}: Start: {convert_to_berlin(start).strftime('%d.%m.%y, %H:%M:%S')}, "
                               f"Dauer: {duration:.2f} Sek., Zuordnung: {assign}\n")


        def calc_cda(air_temp, air_pressure, system_mass, thresh_up, thresh_low, zeit, slope_percent):
            g = 9.81  # m/s^2
            R = 287.05  # J/(kg*K)
            T = air_temp + 273.15  # K
            rho = air_pressure * 100 / (R * T)  # kg/m^3
            m = system_mass
            t = zeit
            a = (thresh_up - thresh_low) / t  # m/s^2
            v_mittel = (thresh_up + thresh_low) / 2  # m/s
            slope_force = - m * g * (slope_percent / 100)

            def angle_deg_to_slope_percent(angle_deg):
                angle_rad = math.radians(angle_deg)
                return math.tan(angle_rad) * 100
            theta = angle_deg_to_slope_percent(slope_percent)

            # Summe der Kräfte: F_Luft = m*a + F_Roll + F_Hang
            # Also: CdA = 2 * (m*a + m*g*crr + m*slope_force) / (rho * v^2)
            # ACHTUNG: slope_force = positive Zahl bei Gefälle!

            cda = (2 * m * a - crr * g * math.cos(theta) - slope_force) / (rho * v_mittel ** 2)

            return cda

        def calc_aero_watts(cda, air_temp, air_pressure, v_mittel):
            R = 287.05  # spezifische Gaskonstante für trockene Luft in J/(kg*K)
            T = air_temp + 273.15  # Umrechnung in Kelvin
            rho = air_pressure*100 / (R * T)  # Luftdichte in kg/m^3
            p_aero = 1/2 * rho * cda * v_mittel ** 3
            return p_aero


        def calc_friction_watts(crr, system_mass, v_mittel):
            p_fric = crr * system_mass * 9.81 * v_mittel
            return p_fric

        def avg_duration(indices):
            values = [events[i - 1][2] for i in indices]
            avg = sum(values) / len(values) if values else 0
            return avg


        def median_duration(indices):
            return np.median([events[i - 1][2] for i in indices]) if indices else 0


        def std_deviation(indices):
            values = [events[i - 1][2] for i in indices]
            std = statistics.stdev(values) if len(values) > 1 else 0
            return std


        def calculate_statistics(indices):
            values = [events[i - 1][2] for i in indices]
            avg = avg_duration(indices)
            std = std_deviation(indices)
            med = median_duration(indices)
            v_mittel = (thresh_up+thresh_low)/2
            selected_method = stat_method_var.get()
            if selected_method == "Arithmetisch":
                try:
                    cda = calc_cda(air_temp, air_pressure, system_mass, thresh_up, thresh_low, avg, slope_percent)
                except:
                    cda = 0
            else:
                try:
                    cda = calc_cda(air_temp, air_pressure, system_mass, thresh_up, thresh_low, med, slope_percent)
                except:
                    cda = 0
            try: p_aero = calc_aero_watts(cda, air_temp, air_pressure, v_mittel)
            except: p_aero = 0
            try: p_fric = calc_friction_watts(crr, system_mass, v_mittel)
            except: p_fric = 0
            return avg, med, std, len(values), cda, v_mittel, p_aero, p_fric


        stats_a = calculate_statistics(set_a_indices)
        stats_b = calculate_statistics(set_b_indices)
        stats_c = calculate_statistics(set_c_indices)
        stats_d = calculate_statistics(set_d_indices)
        if len(set_c_indices) == 0 and len (set_d_indices) == 0:
            result_text = (
                f"\nSet A: {set_a_indices}\nAnzahl: {stats_a[3]}\nØ Dauer: {stats_a[0]:.2f} Sek., Median: {stats_a[1]:.2f} Sek., Std: {stats_a[2]:.2f}\n CdA: {stats_a[4]:.4f}\nBenötigte Leistung für {stats_a[5] * 3.6:.1f} km/h (ohne Gefälle): {stats_a[6] + stats_a[7]:.0f} Watt\n\n"
                f"Set B: {set_b_indices}\nAnzahl: {stats_b[3]}\nØ Dauer: {stats_b[0]:.2f} Sek., Median: {stats_b[1]:.2f} Sek., Std: {stats_b[2]:.2f}\nCdA: {stats_b[4]:.4f}\nBenötigte Leistung für {stats_b[5] * 3.6:.1f} km/h (ohne Gefälle): {stats_b[6] + stats_b[7]:.0f} Watt\n\n"
                f"Nicht zugeordnet: {unassigned_indices}\n"
            )
        else:
            result_text = (
                f"\nSet A: {set_a_indices}\nAnzahl: {stats_a[3]}\nØ Dauer: {stats_a[0]:.2f} Sek., Median: {stats_a[1]:.2f} Sek., Std: {stats_a[2]:.2f}\n CdA: {stats_a[4]:.4f}\nBenötigte Leistung für {stats_a[5]*3.6:.1f} km/h (ohne Gefälle): {stats_a[6] + stats_a[7]:.0f} Watt\n\n"
                f"Set B: {set_b_indices}\nAnzahl: {stats_b[3]}\nØ Dauer: {stats_b[0]:.2f} Sek., Median: {stats_b[1]:.2f} Sek., Std: {stats_b[2]:.2f}\nCdA: {stats_b[4]:.4f}\nBenötigte Leistung für {stats_b[5]*3.6:.1f} km/h (ohne Gefälle): {stats_b[6] + stats_b[7]:.0f} Watt\n\n"
                f"Set C: {set_c_indices}\nAnzahl: {stats_c[3]}\nØ Dauer: {stats_c[0]:.2f} Sek., Median: {stats_c[1]:.2f} Sek., Std: {stats_c[2]:.2f}\nCdA: {stats_c[4]:.4f}\nBenötigte Leistung für {stats_c[5]*3.6:.1f} km/h (ohne Gefälle): {stats_c[6] + stats_c[7]:.0f} Watt\n\n"
                f"Set D: {set_d_indices}\nAnzahl: {stats_d[3]}\nØ Dauer: {stats_d[0]:.2f} Sek., Median: {stats_d[1]:.2f} Sek., Std: {stats_d[2]:.2f}\nCdA: {stats_d[4]:.4f}\nBenötigte Leistung für {stats_d[5]*3.6:.1f} km/h (ohne Gefälle): {stats_d[6] + stats_d[7]:.0f} Watt\n\n"
                f"Nicht zugeordnet: {unassigned_indices}\n"
            )

        out_frame = ttk.Frame(result_frame)
        out_frame.pack(fill="both", expand=True)

        text_box = tk.Text(out_frame, wrap="word", height=10)
        text_box.insert("end", event_list_str + result_text)
        text_box.config(state="disabled")
        text_box.pack(side="left", fill="both", expand=True)

        fig, ax = plt.subplots(figsize=(5, 4))
        plot_speed_events_manual(records, events, set_a_indices, set_b_indices, set_c_indices, set_d_indices, ax)
        canvas.draw()
        canvas.get_tk_widget().pack(side="right", fill="both", expand=True)

    update_display()

# -----------------------------
# Haupt-GUI
# -----------------------------
root = tk.Tk()
root.title("AeroScope v0.3")
root.geometry("800x600")
# Logo laden
try:
    logo_image = Image.open("Logo.png")
    logo_image = logo_image.resize((200, 200))  # Optional: Größe anpassen
except Exception as e:
    logo_photo = None  # Logo nicht verfügbar

# Eingabebereich in einem Grid
input_frame = ttk.Frame(root, padding=10)
input_frame.pack(side="top", fill="x")

# Logo in input_frame einfügen (z. B. rechts oben in Zeile 0, Spalte 10)
logo_label = ttk.Label(input_frame, image=logo_photo)
logo_label.image = logo_photo  # Referenz halten
logo_label.grid(row=0, column=0, rowspan=20, padx=5, pady=2, sticky="ne")

ttk.Label(input_frame, text="Fit-Datei:").grid(row=0, column=1, sticky="w", padx=5, pady=2)
file_entry_widget = ttk.Entry(input_frame, width=20)
file_entry_widget.grid(row=0, column=2, sticky="w", padx=5, pady=2)

def choose_file():
    file_path = filedialog.askopenfilename(filetypes=[("FIT files", "*.fit"), ("Alle Dateien", "*.*")])
    if file_path:
        file_entry_widget.delete(0, tk.END)
        file_entry_widget.insert(0, file_path)

ttk.Button(input_frame, text="Datei auswählen", command=choose_file).grid(row=0, column=3, sticky="w", padx=5, pady=2)

ttk.Label(input_frame, text="Obere Geschwindigkeit [km/h]:").grid(row=1, column=1, sticky="w", padx=5, pady=2)
upper_threshold_entry_widget = ttk.Entry(input_frame, width=10)
upper_threshold_entry_widget.insert(0, "52")
upper_threshold_entry_widget.grid(row=1, column=2, sticky="w", padx=5, pady=2)

ttk.Label(input_frame, text="Untere Geschwindigkeit [km/h]:").grid(row=2, column=1, sticky="w", padx=5, pady=2)
lower_threshold_entry_widget = ttk.Entry(input_frame, width=10)
lower_threshold_entry_widget.insert(0, "49")
lower_threshold_entry_widget.grid(row=2, column=2, sticky="w", padx=5, pady=2)

ttk.Label(input_frame, text="Minimale Dauer [s]:").grid(row=3, column=1, sticky="w", padx=5, pady=2)
min_duration_entry_widget = ttk.Entry(input_frame, width=10)
min_duration_entry_widget.insert(0, "10")
min_duration_entry_widget.grid(row=3, column=2, sticky="w", padx=5, pady=2)

ttk.Label(input_frame, text="Maximale Dauer [s]:").grid(row=4, column=1, sticky="w", padx=5, pady=2)
max_duration_entry_widget = ttk.Entry(input_frame, width=10)
max_duration_entry_widget.insert(0, "30")
max_duration_entry_widget.grid(row=4, column=2, sticky="w", padx=5, pady=2)

ttk.Label(input_frame, text="Runden-Fenster [s]:").grid(row=5, column=1, sticky="w", padx=5, pady=2)
lap_window_entry_widget = ttk.Entry(input_frame, width=10)
lap_window_entry_widget.insert(0, "100")
lap_window_entry_widget.grid(row=5, column=2, sticky="w", padx=5, pady=2)

ttk.Label(input_frame, text="Gesamtmasse System [kg]:").grid(row=1, column=3, sticky="w", padx=5, pady=2)
system_mass_entry_widget = ttk.Entry(input_frame, width=10)
system_mass_entry_widget.insert(0, "90")
system_mass_entry_widget.grid(row=1, column=4, sticky="w", padx=5, pady=2)

ttk.Label(input_frame, text="Mittlerer Umgebungsluftdruck [mbar]:").grid(row=2, column=3, sticky="w", padx=5, pady=2)
air_pressure_entry_widget = ttk.Entry(input_frame, width=10)
air_pressure_entry_widget.insert(0, "1013")
air_pressure_entry_widget.grid(row=2, column=4, sticky="w", padx=5, pady=2)

ttk.Label(input_frame, text="Mittlere Umgebungslufttemperatur [°C]:").grid(row=3, column=3, sticky="w", padx=5, pady=2)
air_temp_entry_widget = ttk.Entry(input_frame, width=10)
air_temp_entry_widget.insert(0, "20")
air_temp_entry_widget.grid(row=3, column=4, sticky="w", padx=5, pady=2)

ttk.Label(input_frame, text="Rollwiderstandskoeffizient (crr):").grid(row=4, column=3, sticky="w", padx=5, pady=2)
crr_entry_widget = ttk.Entry(input_frame, width=10)
crr_entry_widget.insert(0, "0.003")
crr_entry_widget.grid(row=4, column=4, sticky="w", padx=5, pady=2)

ttk.Label(input_frame, text="Gefälle in Prozent").grid(row=5, column=3, sticky="w", padx=5, pady=2)
slope_percent_entry_widget = ttk.Entry(input_frame, width=10)
slope_percent_entry_widget.insert(0, "0")
slope_percent_entry_widget.grid(row=5, column=4, sticky="w", padx=5, pady=2)

ttk.Label(input_frame, text="Glättungsfenster").grid(row=11, column=1, sticky="w", padx=5, pady=2)
filter_window_entry_widget = ttk.Entry(input_frame, width=10)
filter_window_entry_widget.insert(0, "1")
filter_window_entry_widget.grid(row=11, column=2, sticky="w", padx=5, pady=2)

# Auswahl für Mittelwertbildung
ttk.Label(input_frame, text="Mittelwertbildung:").grid(row=12, column=1, sticky="w", padx=5, pady=2)
stat_method_var = tk.StringVar()
stat_method_combobox = ttk.Combobox(input_frame, textvariable=stat_method_var, state="readonly", width=12)
stat_method_combobox['values'] = ("Arithmetisch", "Median")
stat_method_combobox.current(0)  # Standard: Mittelwert
stat_method_combobox.grid(row=12, column=2, sticky="w", padx=5, pady=2)

ttk.Button(input_frame, text="Analyse starten / Reset", command=run_analysis).grid(row=12, column=1, sticky="e", columnspan=3, pady=2)

# Ergebnisbereich
result_frame = ttk.Frame(root, padding=10)
result_frame.pack(side="top", fill="both", expand=True)

root.mainloop()

def run_analysis(
    filepath,
    upper_threshold, lower_threshold,
    min_duration, max_duration,
    lap_window_val,
    system_mass,
    air_pressure, air_temp, crr
):
    """
    Liest die FIT-Datei ein, führt alle Berechnungen aus
    und liefert:
      - figures: Liste von (Titel, Base64-String)-Tuples
      - stats: Dict mit allen berechneten Werten
    """

    # Beispiel: so liest du die FIT und sammelst Daten
    fit = FitFile(filepath)
    records = []
    for rec in fit.get_messages("record"):
        data = {f.name: f.value for f in rec}
        if "timestamp" in data and "heart_rate" in data:
            records.append((data["timestamp"], data["heart_rate"]))

    # ────────────── HIER rufst du deine eigenen Plot-Funktionen auf ──────────────
    figures = []

    # Beispiel-Plot (du ersetzt das durch deine eigenen):
    fig, ax = plt.subplots()
    times = [t for t, _ in records]
    vals  = [v for _, v in records]
    ax.plot(times, vals)
    ax.set_title("Herzfrequenz über Zeit")
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Herzfrequenz (bpm)")

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("ascii")
    figures.append(("Herzfrequenz über Zeit", img_b64))

    # ────────────── STATISTIK-BERECHNUNGEN ─────────────────────────────────────
    stats = {
        "Anzahl Datensätze": len(records),
        "Durchschnitt HF": statistics.mean(vals) if vals else None,
        "Median HF":    median(vals)     if vals else None,
        "Max HF":       max(vals)        if vals else None,
        "Min HF":       min(vals)        if vals else None,
    }

    return figures, stats