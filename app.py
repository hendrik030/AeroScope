# app.py
import os
from flask import Flask, render_template, request, redirect, url_for
from aeroscope_logic import parse_fit, plot_to_base64
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "/tmp/uploads"
ALLOWED_EXT = {"fit"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXT

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        f = request.files.get("file")
        if f and allowed(f.filename):
            fn = secure_filename(f.filename)
            path = os.path.join(app.config["UPLOAD_FOLDER"], fn)
            f.save(path)
            times, hr = parse_fit(path)
            img = plot_to_base64(times, hr)
            return render_template("index.html", img_data=img)
        else:
            return render_template("index.html", error="Bitte FIT-Datei hochladen.")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))