# Augmentation Playground

**Interactive browser UI for building [Albumentations](https://albumentations.ai/) pipelines — tweak sliders, see results live, copy the exact Python code.**

> Upload an image, drag a slider, hit apply. The generated `A.Compose([...])` snippet in the right panel always matches what you see.

![demo](docs/demo.gif)

---

## Why this exists

Building a good augmentation pipeline usually means writing code, running training, checking if it helped, adjusting, repeating. This tool collapses that loop — you see the effect of every parameter *before* you commit it to your training script.

It also supports **full polygon labels** (YOLO segmentation format) — upload a `.txt` label file alongside your image and the polygon contours transform correctly with every augmentation, so you can verify your geometric transforms aren't breaking your annotations.

---

## Features

| | |
|---|---|
| **Side-by-side compare** | Original and augmented shown at the same size, same position |
| **Grid ×9** | See 9 different random variants at once — great for judging augmentation strength |
| **Polygon label support** | Upload YOLO polygon `.txt` labels, contours drawn and correctly transformed |
| **Toggle overlay** | Switch polygon visualization on/off without re-running augmentation |
| **Code export** | Generated `A.Compose([...])` snippet matches your exact slider settings, one-click copy |
| **URL sharing** | All settings encoded in the URL hash — share your pipeline config with a link |
| **Seed control** | Lock a seed for reproducible results or shuffle for exploration |
| **Keyboard shortcuts** | `Enter` to apply, `R` to shuffle seed |

---

## Quick start

```bash
# Docker — zero setup
git clone https://github.com/YOUR_USERNAME/augmentation-playground
cd augmentation-playground
docker compose up
```

Open **http://localhost:8000**

```bash
# Or plain Python
pip install -r requirements.txt
python main.py
```

---

## Augmentations

| Category | Transforms |
|---|---|
| Geometric | `HorizontalFlip` `VerticalFlip` `RandomRotate90` `Rotate` `RandomScale` |
| Color / exposure | `RandomBrightnessContrast` `HueSaturationValue` `RGBShift` `CLAHE` `Equalize` |
| Noise | `GaussNoise` |
| Blur | `GaussianBlur` `MotionBlur` |
| Dropout | `CoarseDropout` (cutout) |

---

## Label format

Expects YOLO polygon / segmentation format — the same format output by tools like Roboflow, CVAT, and Label Studio:

```
class_id  x1 y1  x2 y2  x3 y3  ...  xN yN
```

Coordinates are normalized `[0, 1]`. Any number of points per grain. Multiple grains per file.

---

## REST API

The backend is plain FastAPI — use it directly from your training scripts:

```python
import requests, base64

with open("image.jpg", "rb") as f:
    b64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

with open("image.txt") as f:
    labels = f.read()

res = requests.post("http://localhost:8000/augment", json={
    "image_b64": b64,
    "label_text": labels,
    "params": {
        "hflip": True,
        "rotate_limit": 30,
        "brightness_limit": 0.4,
        "gauss_noise": 0.05,
        "seed": 42,
    }
})

data = res.json()
# data["aug_clean_b64"]      — augmented image (clean)
# data["aug_annot_b64"]      — augmented image with polygons drawn
# data["aug_polygon_count"]  — how many polygons survived the transform
# data["code"]               — ready-to-paste Albumentations snippet
```

| Endpoint | Description |
|---|---|
| `POST /upload` | Upload image file → base64 |
| `POST /preview_labels` | Draw polygons on original image |
| `POST /augment` | Augment single image, returns clean + annotated versions |
| `POST /augment_grid` | Augment N variants, returns clean + annotated grids |

---

## Stack

- **Backend**: Python · FastAPI · Albumentations · OpenCV — `main.py` is ~300 lines
- **Frontend**: single `index.html` — no React, no build step, no Node
- **Deploy**: `docker compose up`

---

## Contributing

PRs welcome. Most wanted:

- [ ] Live update on slider drag (debounced, no apply button needed)
- [ ] Export augmented dataset (apply pipeline to folder of images)
- [ ] More transforms (ElasticTransform, GridDistortion, MixUp)
- [ ] Drag-and-drop reordering of transforms

---

## License

MIT
