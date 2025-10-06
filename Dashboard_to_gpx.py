import cv2
import pytesseract
import re
import os
from tkinter import Tk, filedialog
from datetime import datetime, timedelta
from PIL import Image
import numpy as np
import time
import sys
from math import radians, sin, cos, sqrt, atan2

# If Tesseract not in PATH, uncomment this line:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

TESS_CONFIG = r'--psm 6 -c tessedit_char_whitelist=0123456789NnSsEeWw°.,-'

# ---------------- OCR + coordinate parsing ----------------

def ocr_text(img):
    return pytesseract.image_to_string(Image.fromarray(img), config=TESS_CONFIG)

def extract_coords_from_text(text):
    # Match typical GPS like: 43.1234N 79.1234W or -43.123, 79.123
    pattern = re.compile(r"(-?\d{1,3}\.\d+)[ ,°]*([NnSs])?[^0-9\-]*(-?\d{1,3}\.\d+)[ ,°]*([EeWw])?")
    match = pattern.search(text)
    if match:
        lat = float(match.group(1))
        lon = float(match.group(3))
        ns = match.group(2)
        ew = match.group(4)
        if ns:
            lat = -abs(lat) if ns.upper() == "S" else abs(lat)
        if ew:
            lon = -abs(lon) if ew.upper() == "W" else abs(lon)
        return lat, lon
    return None

# ---------------- Sanity check & correction ----------------

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2.0)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2.0)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def correct_sign(lat, lon, expected_lat_sign, expected_lon_sign):
    flipped = [False, False]
    if abs(lat) > 1 and (lat * expected_lat_sign < 0):
        lat = -lat
        flipped[0] = True
    if abs(lon) > 1 and (lon * expected_lon_sign < 0):
        lon = -lon
        flipped[1] = True
    return lat, lon, flipped

def sanity_check(points, expected_lat_sign, expected_lon_sign):
    cleaned = []
    flips = 0
    skipped = 0
    for i, (t, lat, lon) in enumerate(points):
        lat, lon, flipped = correct_sign(lat, lon, expected_lat_sign, expected_lon_sign)
        if any(flipped):
            flips += 1
        if i > 0 and cleaned:
            prev_t, prev_lat, prev_lon = cleaned[-1]
            dt = (t - prev_t).total_seconds()
            if dt > 0:
                dist = haversine(prev_lat, prev_lon, lat, lon)
                speed = dist / dt * 3.6
                if speed > 300:  # unrealistic jump
                    skipped += 1
                    continue
        cleaned.append((t, lat, lon))
    return cleaned, flips, skipped

# ---------------- GPX writer ----------------

def write_gpx(points, out_file):
    with open(out_file, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<gpx version="1.1" creator="DashcamExtractor" xmlns="http://www.topografix.com/GPX/1/1">\n')
        f.write('  <trk>\n    <trkseg>\n')
        for t, lat, lon in points:
            f.write(f'      <trkpt lat="{lat:.6f}" lon="{lon:.6f}"><time>{t.isoformat()}Z</time></trkpt>\n')
        f.write('    </trkseg>\n  </trk>\n</gpx>\n')

# ---------------- Video processing ----------------

def extract_coords_from_video(video_path, start_dt, out_gpx, lat_sign, lon_sign):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    print(f"Video duration: {duration:.1f}s | FPS: {fps:.2f}")

    # ROI = bottom 9% of the frame
    ret, frame = cap.read()
    if not ret:
        print("Could not read video frame.")
        return
    h, w, _ = frame.shape
    y1 = int(h * 0.91)
    y2 = h
    x1 = int(w * 0.32)
    x2 = int(w * 0.54)

    print("\nROI confirmed.")
    time.sleep(1)

    points = []
    last_time = start_dt
    print("\n[STEP 1] Extracting coordinates…")

    for sec in range(int(duration)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
        ret, frame = cap.read()
        if not ret:
            break
        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        text = ocr_text(thresh)
        coords = extract_coords_from_text(text)
        if coords:
            lat, lon = coords
            timestamp = start_dt + timedelta(seconds=sec)
            points.append((timestamp, lat, lon))
            last_time = timestamp

        pct = sec / duration * 100
        eta = duration - sec
        sys.stdout.write(f"\r {pct:5.1f}% | video time {str(last_time.time())} | ETA {int(eta)}s")
        sys.stdout.flush()

    cap.release()
    print("\n\n[STEP 2] Sanity check…")
    cleaned_points, flips, skipped = sanity_check(points, lat_sign, lon_sign)

    print(f"[STEP 3] Writing GPX with {len(cleaned_points)} points…")
    write_gpx(cleaned_points, out_gpx)

    # Summary
    total = len(points)
    print("\n✅ GPX saved to", out_gpx)
    print(f"--- SUMMARY ---")
    print(f"Total extracted points: {total}")
    print(f"Sign corrections:       {flips}")
    print(f"Skipped (bad jumps):    {skipped}")
    print(f"Final kept points:      {len(cleaned_points)}")
    print("----------------")

# ---------------- Main ----------------

def main():
    Tk().withdraw()
    video_path = filedialog.askopenfilename(title="Select dashcam video", filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if not video_path:
        print("No video selected.")
        return

    start_dt_str = input("Enter start datetime (YYYY-mm-dd HH:MM:SS): ")
    start_dt = datetime.strptime(start_dt_str, "%Y-%m-%d %H:%M:%S")

    lat_input = input("Is latitude South? (y/n): ").strip().lower()
    lon_input = input("Is longitude West? (y/n): ").strip().lower()
    lat_sign = -1 if lat_input == "y" else 1
    lon_sign = -1 if lon_input == "y" else 1

    out_gpx = os.path.splitext(video_path)[0] + ".gpx"
    extract_coords_from_video(video_path, start_dt, out_gpx, lat_sign, lon_sign)

if __name__ == "__main__":
    main()
