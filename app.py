# app.py

import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from aeroscope_logic import run_analysis

# Konfiguration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXT = {"fit"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def allowed(fname):
    return "." in fname and fname.rsplit(".",1)[1].lower() in ALLOWED_EXT

@app.route("/", methods=["GET","POST"])
def index():
    error = None
    figures = None
    stats   = None
    if request.method == "POST":
        f = request.files.get("file")
        if not f or not allowed(f.filename):
            error = "Bitte eine .fit-Datei auswählen."
        else:
            fn   = secure_filename(f.filename)
            path = os.path.join(app.config["UPLOAD_FOLDER"], fn)
            f.save(path)
            # Hier übernimmst du dieselben Parameter wie im Desktop-Tool:
            figures, stats = run_analysis(
                filepath=path,
                upper_threshold=float(request.form.get("upper_threshold", 5)),
                lower_threshold=float(request.form.get("lower_threshold", 1)),
                min_duration=float(request.form.get("min_duration", 0.5)),
                max_duration=float(request.form.get("max_duration", 10)),
                lap_window_val=float(request.form.get("lap_window", 10)),
                system_mass=float(request.form.get("system_mass", 75)),
                air_pressure=float(request.form.get("air_pressure", 1013)),
                air_temp=float(request.form.get("air_temp", 20)),
                crr=float(request.form.get("crr", 0.005))
            )
    return render_template(
        "index.html",
        error=error,
        figures=figures,
        stats=stats
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))