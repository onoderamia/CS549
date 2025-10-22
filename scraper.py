import os, random, requests
from pathlib import Path
from io import BytesIO
from PIL import Image

API_KEY = "" 

REGIONS = {
    "North America":   {"city": "Chicago", "lat": 41.8781, "lng": -87.6298},
    "South America":   {"city": "Bogota",  "lat": 4.7110,  "lng": -74.0721},
    "Western Europe":  {"city": "London",  "lat": 51.5074, "lng": -0.1278},
    "Southern Europe": {"city": "Rome",    "lat": 41.9028, "lng": 12.4964},
    "Middle East":     {"city": "Dubai",   "lat": 25.276987, "lng": 55.296249},
    "Sub-Saharan Africa": {"city": "Lagos", "lat": 6.5244, "lng": 3.3792},
    "South Asia":      {"city": "Delhi",   "lat": 28.6139, "lng": 77.2090},
    "East Asia":       {"city": "Tokyo",   "lat": 35.6762, "lng": 139.6503},
    "Southeast Asia":  {"city": "Hanoi",   "lat": 21.0285, "lng": 105.8542},
}

META_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"
OUT = Path("/Users/miaonodera/Desktop/UIUC/FALL2025/ECE549/CV/test_out")
OUT.mkdir(exist_ok=True)

def streetview_meta(lat, lng, radius=60):
    params = {"location": f"{lat},{lng}", "radius": radius, "source": "outdoor", "key": API_KEY}
    r = requests.get(META_URL, params=params)
    if r.status_code != 200:
        return None
    return r.json()

def streetview_image(pano_id, heading=0, fov=80, pitch=0):
    params = {
        "pano": pano_id,
        "size": "640x640",
        "heading": heading,
        "pitch": pitch,
        "fov": fov,
        "key": API_KEY,
    }
    r = requests.get(IMAGE_URL, params=params)
    if r.status_code != 200:
        return None
    return Image.open(BytesIO(r.content))

def main():
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        raise SystemExit("Set GCP_API_KEY environment variable or paste your key into API_KEY.")

    sample_regions = random.sample(list(REGIONS.keys()), 3)
    for region in sample_regions:
        info = REGIONS[region]
        lat, lng = info["lat"], info["lng"]
        print(f"\n{region} ({info['city']})")
        meta = streetview_meta(lat, lng)
        if not meta or meta.get("status") != "OK":
            print("  No Street View found nearby.")
            continue

        pano_id = meta["pano_id"]
        img = streetview_image(pano_id, heading=random.choice([0, 90, 180, 270]))
        if img:
            out_path = OUT / f"{region.replace(' ','_')}_{info['city']}.jpg"
            img.save(out_path)
            print(f"  Saved {out_path}")
        else:
            print("  Failed to fetch image.")

    print("\nDone. Check the 'test_out' folder for images.")

if __name__ == "__main__":
    main()
