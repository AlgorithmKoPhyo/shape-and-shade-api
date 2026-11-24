import json
from pathlib import Path
import httpx

BASE_URL = "http://127.0.0.1:8000"
IMAGE_PATH = Path("tests/assets/lena.jpg")
OUTPUT_PATH = Path("extracted_matrices.json")

with httpx.Client(timeout=30.0) as client:
    # 1. ပုံတင်ပြီး RGB/LAB matrices ဆွဲထုတ်မယ်
    with IMAGE_PATH.open("rb") as f:
        res = client.post(
            f"{BASE_URL}/api/v1/extract-skin-regions",
            files={"file": (IMAGE_PATH.name, f, "image/jpeg")},
        )
    res.raise_for_status()
    data = res.json()

    # 2. JSON ဖိုင်အဖြစ် သိမ်းမယ်
    OUTPUT_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved full response to: {OUTPUT_PATH}")

    # 3. Region တစ်ခုချင်းစီရဲ့ pixel အရေအတွက်နဲ့ နမူနာ value စစ်ကြည့်မယ်
    for region_name, patch in data["region_patches"].items():
        rgb_len = len(patch["rgb_pixels"])
        lab_len = len(patch["lab_pixels"])
        print(
            f"- {region_name:12s} | RGB pixels: {rgb_len} (sample: {patch['rgb_pixels'][0]}) "
            f"| LAB pixels: {lab_len} (sample: {patch['lab_pixels'][0]})"
        )

    # 4. /api/v1/history endpoint ကိုပါ လှမ်းစစ်မယ်
    hist_res = client.get(f"{BASE_URL}/api/v1/history")
    print(f"\nHistory count in server memory: {hist_res.json()['count']}")