import os, random, requests
from pathlib import Path
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from dotenv import load_dotenv

load_dotenv(".env")
API_KEY = os.getenv("API_KEY")

BBOXES = {
    "Chicago": (41.8650, 41.9010, -87.6650, -87.6000),
    "Bogota":  (4.5900,  4.7200,  -74.1100, -74.0300),
    "London":  (51.4900, 51.5400, -0.1600,   -0.0700),
    "Rome":    (41.8800, 41.9100,  12.4600,  12.5200),
    "Dubai":   (25.1800, 25.2300,  55.2500,  55.3100),
    "Lagos":   (6.4200,  6.4600,    3.3800,   3.4300),
    "Delhi":   (28.6100, 28.6400,  77.2000,  77.2500),
    "Tokyo":   (35.6750, 35.7050, 139.7300, 139.7700),
    "Hanoi":   (21.0200, 21.0450, 105.8300, 105.8600),
}

META_URL  = "https://maps.googleapis.com/maps/api/streetview/metadata"
IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"
OUT = Path("out"); OUT.mkdir(exist_ok=True)

def rand_point(bbox):
    lat0, lat1, lng0, lng1 = bbox
    return random.uniform(lat0, lat1), random.uniform(lng0, lng1)

def get_meta(lat, lng, radius=120):
    p = {"location": f"{lat},{lng}", "radius": radius, "source": "outdoor", "key": API_KEY}
    r = requests.get(META_URL, params=p, timeout=15)
    if r.status_code != 200: return None
    j = r.json()
    return j if j.get("status") == "OK" and j.get("pano_id") else None

def get_image(pano_id, heading, fov=80, pitch=-10):
    p = {"pano": pano_id, "size": "640x640", "heading": heading, "pitch": pitch, "fov": fov, "key": API_KEY}
    r = requests.get(IMAGE_URL, params=p, timeout=20)
    if r.status_code != 200: return None
    try:
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        return None

def score_streetness(pil_img):
    img_rgb = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # sharpness 
    lap_var = cv2.Laplacian(gray, cv2.CV_64F, ksize=3).var()
    sharp = min(lap_var, 300.0) / 300.0

    # edges lower vs upper
    edges = cv2.Canny(gray, 60, 160, L2gradient=True)
    h, w = edges.shape
    lower = edges[h//2:, :]
    upper = edges[:h//2, :]
    ed_lo   = (lower.mean() / 255.0) * 3.0
    ed_diff = max((lower.mean() - upper.mean()) / 255.0, 0) * 3.0
    ed_lo   = min(ed_lo, 1.0)
    ed_diff = min(ed_diff, 1.0)

    # wall
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    ax = np.mean(np.abs(sobelx))
    ay = np.mean(np.abs(sobely))
    vert_ratio = ax / (ax + ay + 1e-6)      
    over_vert  = max(vert_ratio - 0.60, 0.0)   
    vert_penalty = 1.0 - min(over_vert * 0.7, 0.5)

    score = (0.45 * sharp) + (0.30 * ed_lo) + (0.25 * ed_diff)
    score *= vert_penalty
    return float(max(0.0, min(1.0, score)))

def is_meaningful(pil_img, thr=0.5):
    return score_streetness(pil_img) >= thr

def main():
    if not API_KEY or API_KEY.strip() == "YOUR_GOOGLE_API_KEY_HERE":
        raise SystemExit("Set API_KEY to your Google Street View Static API key.")
    
    HEADINGS = [0, 90, 180, 270]
    TRIES_PER_CITY = 5
    META_RADIUS = 120
    FALLBACK_RADIUS = 180

    for city, bbox in BBOXES.items():
        city_dir = OUT / city
        city_dir.mkdir(exist_ok=True)

        saved = False
        for _ in range(TRIES_PER_CITY):
            lat, lng = rand_point(bbox)
            meta = get_meta(lat, lng, META_RADIUS) or get_meta(lat, lng, FALLBACK_RADIUS)
            if not meta: 
                continue

            pid = meta["pano_id"]
            best = (-1.0, None, None) 

            for hdg in random.sample(HEADINGS, k=len(HEADINGS)):
                img = get_image(pid, hdg)
                if img is None: 
                    continue
                sc = score_streetness(img)
                if sc > best[0]:
                    best = (sc, img, hdg)

            if best[1] is not None and is_meaningful(best[1]):
                fname = f"{lat:.5f},{lng:.5f}.jpg"
                best[1].save(city_dir / fname, quality=92)
                print(f"[{city}] saved {fname}")
                saved = True
                break

    print(f"done. check: {OUT.resolve()}")

if __name__ == "__main__":
    main()
