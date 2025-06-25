# aeroscope_logic.py

import io
import base64
import datetime
import statistics
import math
from statistics import median
from datetime import timedelta
from zoneinfo import ZoneInfo
import numpy as np
import matplotlib.pyplot as plt
from fitparse import FitFile
import mplcursors

# -----------------------------
# Helper Functions
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

# -----------------------------
# Plot Function
# -----------------------------
def plot_speed_events_manual(records, events, set_a_indices, set_b_indices, set_c_indices, set_d_indices, ax):
    ax.clear()
    set_colors = {'A': 'green', 'B': 'orange', 'C': 'purple', 'D': 'red'}
    times = [t for (t, _) in records]
    speeds = [v * 3.6 for (_, v) in records]  # m/s to km/h

    plotted_lines = []
    event_labels = []

    def plot_event_set(indices, color):
        for idx in indices:
            start, end, duration = events[idx-1]
            event_times = [t for t in times if start <= t <= end]
            if not event_times:
                continue
            start_i = times.index(event_times[0])
            end_i = times.index(event_times[-1])
            ev_speeds = speeds[start_i:end_i+1]
            rel_times = [(t - start).total_seconds() for t in event_times]
            line, = ax.plot(rel_times, ev_speeds, color=color)
            plotted_lines.append(line)
            event_labels.append(f"Event {idx}")
            ax.text(0, ev_speeds[0], str(idx), fontsize=10, color=color, ha='left', va='bottom')
            ax.text(rel_times[-1], ev_speeds[-1], str(idx), fontsize=10, color=color, ha='right', va='top')

    plot_event_set(set_a_indices, set_colors['A'])
    plot_event_set(set_b_indices, set_colors['B'])
    plot_event_set(set_c_indices, set_colors['C'])
    plot_event_set(set_d_indices, set_colors['D'])

    legend_elements = [plt.Line2D([0],[0], color=c, lw=4) for c in set_colors.values()]
    ax.legend(legend_elements, ['Set A','Set B','Set C','Set D'], loc='upper right')
    ax.set_xlabel("Zeit (s)")
    ax.set_ylabel("Geschwindigkeit (km/h)")
    ax.grid(True)

    cursor = mplcursors.cursor(plotted_lines, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        sel.annotation.set_text(event_labels[plotted_lines.index(sel.artist)])

# -----------------------------
# Main Analysis Function
# -----------------------------
def run_analysis(
    filepath,
    upper_threshold, lower_threshold,
    min_duration, max_duration,
    lap_window_val,
    system_mass,
    air_pressure, air_temp, crr
):
    # Read FIT file
    fit = FitFile(filepath)
    records = []
    for rec in fit.get_messages("record"):
        data = {f.name: f.value for f in rec}
        if "timestamp" in data and "speed" in data and data["speed"] is not None:
            records.append((data["timestamp"], data["speed"]))
    records.sort(key=lambda x: x[0])

    # Smooth speeds if needed
    if len(records) >= 1:
        times, speeds = zip(*records)
        speeds = np.array(speeds)
        if lap_window_val > 1:
            weights = np.ones(int(lap_window_val))/lap_window_val
            sm = np.convolve(speeds, weights, mode='same')
            records = list(zip(times, sm))
    # Detect events
    events = []
    thresh_up = upper_threshold/3.6
    thresh_low = lower_threshold/3.6
    event_start = None
    for i in range(len(records)-1):
        t1,v1 = records[i]
        t2,v2 = records[i+1]
        if event_start is None and v1>=thresh_up and v2<thresh_up:
            event_start = interpolate_time(t1,t2,v1,v2,thresh_up)
        elif event_start and v1>thresh_low and v2<=thresh_low:
            event_end = interpolate_time(t1,t2,v1,v2,thresh_low)
            duration = (event_end-event_start).total_seconds()
            if min_duration<=duration<=max_duration:
                events.append((event_start,event_end,duration))
            event_start = None

    # Assign default sets
    count = len(events)
    set_a = list(range(1, count+1))
    set_b = set_c = set_d = []

    # Generate plot
    figures = []
    fig, ax = plt.subplots()
    plot_speed_events_manual(records, events, set_a, set_b, set_c, set_d, ax)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('ascii')
    figures.append(('Geschwindigkeit & Events', img))

    # Basic stats
    durations = [e[2] for e in events]
    stats = {
        'Anzahl Ereignisse': len(events),
        'Durchschnitt Dauer (s)': statistics.mean(durations) if durations else 0,
        'Max Dauer (s)': max(durations) if durations else 0,
        'Min Dauer (s)': min(durations) if durations else 0,
    }
    return figures, stats