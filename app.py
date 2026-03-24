from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import numpy as np
import threading
import time
import random  # Replace with real serial + model logic
import joblib
from scipy.fft import fft
import pandas as pd
import serial
model = joblib.load("faultdetect.pkl")
scaler = joblib.load("scaler.pkl")


# ==============================
# CONFIG
# ==============================
PORT = "COM5"       # CHANGE to your port
BAUD = 115200
WINDOW_SIZE = 128
RMS_TRIGGER = 0.015

app = Flask(__name__)
CORS(app)

# ==============================
# SHARED STATE
# ==============================
state = {
    "status": "IDLE",
    "rms": 0.0,
    "history": [],          # list of {time, rms, status}
    "fault_count": 0,
    "normal_count": 0,
    "total_windows": 0,
    "running": False,
    "last_features": {}
}
state_lock = threading.Lock()



def extract_features(signal):
    from scipy.fft import fft as scipy_fft
    import pandas as pd
    sig = np.array(signal)
    fft_vals = np.abs(scipy_fft(sig))[:len(sig)//2]
    return [
        float(np.mean(sig)),
        float(np.std(sig)),
        float(np.sqrt(np.mean(sig**2))),
        float(np.max(sig)),
        float(np.min(sig)),
        float(pd.Series(sig).kurtosis()),
        float(pd.Series(sig).skew()),
        float(np.mean(fft_vals)),
        float(np.max(fft_vals)),
    ]



def simulate_vibration():
    import serial
    ser = serial.Serial(PORT, BAUD, timeout=1)
    buffer = []

    while state["running"]:
        try:
            line = ser.readline().decode().strip()

            if not line:
                continue  # skip empty lines

            ax, ay, az = map(float, line.split(","))

            vibration = np.sqrt(ax**2 + ay**2 + az**2)
            buffer.append(vibration)

            if len(buffer) >= WINDOW_SIZE:
                window = np.array(buffer[:WINDOW_SIZE])
                buffer = []

                window = window - np.mean(window)
                rms = float(np.sqrt(np.mean(window**2)))

                if rms < RMS_TRIGGER:
                    prediction = "NORMAL"
                else:
                    features = extract_features(window)
                    features_scaled = scaler.transform([features])
                    prediction = model.predict(features_scaled)[0].upper()
                    if prediction=="WEAR":
                        prediction="FAULT"

                with state_lock:
                    state["status"] = prediction
                    state["rms"] = round(rms, 6)
                    state["total_windows"] += 1

                    if prediction == "FAULT":
                        state["fault_count"] += 1
                    else:
                        state["normal_count"] += 1

    # ✅ HISTORY (timeline + log)
                    state["history"].append({
                        "time": time.strftime("%H:%M:%S"),
                        "rms": round(rms, 6),
                        "status": prediction
                    })

                    if len(state["history"]) > 60:
                        state["history"] = state["history"][-60:]

    # ✅ FEATURES (right panel)
                    features = extract_features(window)
                    state["last_features"] = {
                        "mean": round(features[0], 6),
                        "std": round(features[1], 6),
                        "rms": round(features[2], 6),
                        "max": round(features[3], 6),
                        "min": round(features[4], 6),
                        "kurtosis": round(features[5], 6),
                        "skewness": round(features[6], 6),
                        "fft_mean": round(features[7], 6),
                        "fft_max": round(features[8], 6),
                    }

        except ValueError:
            # Happens when split/map fails (bad serial line)
            continue

        except Exception as e:
            print("Error:", e)
            continue

# ==============================
# ROUTES
# ==============================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def get_status():
    with state_lock:
        return jsonify({
            "status": state["status"],
            "rms": state["rms"],
            "fault_count": state["fault_count"],
            "normal_count": state["normal_count"],
            "total_windows": state["total_windows"],
            "running": state["running"],
            "last_features": state["last_features"],
            "history": state["history"][-20:]
        })

@app.route("/api/start", methods=["POST"])
def start():
    with state_lock:
        if state["running"]:
            return jsonify({"message": "Already running"}), 400
        state["running"] = True
    t = threading.Thread(target=simulate_vibration, daemon=True)
    t.start()
    return jsonify({"message": "Started"})

@app.route("/api/stop", methods=["POST"])
def stop():
    with state_lock:
        state["running"] = False
        state["status"] = "IDLE"
    return jsonify({"message": "Stopped"})

@app.route("/api/reset", methods=["POST"])
def reset():
    with state_lock:
        state["fault_count"] = 0
        state["normal_count"] = 0
        state["total_windows"] = 0
        state["history"] = []
        state["status"] = "IDLE"
        state["rms"] = 0.0
    return jsonify({"message": "Reset"})

@app.route("/api/config", methods=["POST"])
def set_config():
    global RMS_TRIGGER
    data = request.json
    if "rms_trigger" in data:
        RMS_TRIGGER = float(data["rms_trigger"])
    return jsonify({"message": "Config updated", "rms_trigger": RMS_TRIGGER})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
