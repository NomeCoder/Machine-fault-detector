import serial
import numpy as np
import joblib
from scipy.fft import fft
import pandas as pd
import time
# ==============================
# CONFIG
# ==============================
PORT = "COM5"   # CHANGE
BAUD = 115200
WINDOW_SIZE = 128
RMS_TRIGGER = 0.015   # sensitivity (adjust)

# ==============================
# LOAD MODEL
# ==============================
model = joblib.load("faultdetect.pkl")
scaler = joblib.load("scaler.pkl")

ser = serial.Serial(PORT, BAUD, timeout=1)
buffer = []

print("RMS-Based ML Detection Started...")

def extract_features(signal):
    features = [
        np.mean(signal),
        np.std(signal),
        np.sqrt(np.mean(signal**2)),
        np.max(signal),
        np.min(signal),
        pd.Series(signal).kurtosis(),
        pd.Series(signal).skew(),
        np.mean(np.abs(fft(signal))[:len(signal)//2]),
        np.max(np.abs(fft(signal))[:len(signal)//2]),
    ]
    return features

while True:
    try:
        line = ser.readline().decode().strip()
        ax, ay, az = map(float, line.split(","))

        vibration = np.sqrt(ax**2 + ay**2 + az**2)
        buffer.append(vibration)

        if len(buffer) >= WINDOW_SIZE:

            window = np.array(buffer[:WINDOW_SIZE])
            buffer = []

            window = window - np.mean(window)

            rms = np.sqrt(np.mean(window**2))
            print("RMS:", round(rms, 5))

            # ===== Trigger ML only if vibration strong =====
            if rms < RMS_TRIGGER:
                print("STATUS: NORMAL")
                time.sleep(0.01)
                continue

            features = extract_features(window)
            features_scaled = scaler.transform([features])

            prediction = model.predict(features_scaled)[0]
            print("STATUS:", prediction.upper())
            time.sleep(0.01)



    except:
        pass