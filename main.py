"""
Augmentation Playground — FastAPI backend

Label format: class_id x1 y1 x2 y2 ... xN yN  (normalized 0-1, any number of points)
"""

import io
import base64
import random

import cv2
import numpy as np
from PIL import Image
import albumentations as A

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Augmentation Playground")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ─────────────────────────────────────────────────────────────
#  Health check
# ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Liveness / readiness probe for Docker and container orchestrators."""
    return {"status": "ok"}


# ─────────────────────────────────────────────────────────────
#  Models
# ─────────────────────────────────────────────────────────────

class AugParams(BaseModel):
    hflip: bool = True
    vflip: bool = False
    rot90: bool = True
    rotate_limit: int = 25
    rotate_p: float = 0.7
    scale_min: float = 0.8
    scale_max: float = 1.2
    scale_p: float = 0.5
    brightness_limit: float = 0.4
    contrast_limit: float = 0.35
    brightness_p: float = 0.7
    hue_shift: int = 25
    sat_shift: int = 35
    val_shift: int = 30
    hsv_p: float = 0.6
    rgb_shift: int = 30
    rgb_p: float = 0.4
    clahe: bool = True
    clahe_p: float = 0.35
    equalize: bool = False
    equalize_p: float = 0.2
    gauss_noise: float = 0.08
    gauss_noise_p: float = 0.4
    motion_blur: bool = False
    motion_blur_p: float = 0.3
    gaussian_blur: bool = False
    gaussian_blur_p: float = 0.3
    cutout: bool = False
    cutout_p: float = 0.3
    seed: int = 42
    grid_n: int = 1


class AugRequest(BaseModel):
    image_b64: str
    params: AugParams
    label_text: str = ""   # raw label file content, one line per grain


# ─────────────────────────────────────────────────────────────
#  Polygon parsing
# ─────────────────────────────────────────────────────────────

def parse_polygon_labels(label_text: str) -> list:
    polygons = []
    for line in label_text.strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        coords = parts[1:]
        if len(coords) % 2 != 0:
            coords = coords[:-1]
        pts = []
        for i in range(0, len(coords), 2):
            x = min(max(float(coords[i]),   0.0), 1.0)
            y = min(max(float(coords[i+1]), 0.0), 1.0)
            pts.append((x, y))
        if len(pts) >= 3:
            polygons.append(pts)
    return polygons


def polygons_to_keypoints(polygons):
    keypoints, grain_sizes = [], []
    for poly in polygons:
        grain_sizes.append(len(poly))
        keypoints.extend(poly)
    return keypoints, grain_sizes


def keypoints_to_polygons(keypoints, grain_sizes):
    polygons, idx = [], 0
    for size in grain_sizes:
        pts = [p for p in keypoints[idx: idx + size] if p is not None]
        if len(pts) >= 3:
            polygons.append(pts)
        idx += size
    return polygons


# ─────────────────────────────────────────────────────────────
#  Augmentation pipeline
# ─────────────────────────────────────────────────────────────

def build_transform(p: AugParams, has_polygons: bool = False) -> A.Compose:
    transforms = []
    if p.hflip:
        transforms.append(A.HorizontalFlip(p=0.5))
    if p.vflip:
        transforms.append(A.VerticalFlip(p=0.5))
    if p.rot90:
        transforms.append(A.RandomRotate90(p=0.5))
    if p.rotate_limit > 0:
        transforms.append(A.Rotate(limit=p.rotate_limit, p=p.rotate_p))
    if p.scale_min != 1.0 or p.scale_max != 1.0:
        transforms.append(A.RandomScale(
            scale_limit=(p.scale_min - 1.0, p.scale_max - 1.0), p=p.scale_p))

    color_group = []
    if p.brightness_limit > 0:
        color_group.append(A.RandomBrightnessContrast(
            brightness_limit=p.brightness_limit, contrast_limit=p.contrast_limit, p=1.0))
    if p.hue_shift > 0:
        color_group.append(A.HueSaturationValue(
            hue_shift_limit=p.hue_shift, sat_shift_limit=p.sat_shift,
            val_shift_limit=p.val_shift, p=1.0))
    if color_group:
        transforms.append(A.OneOf(color_group, p=p.brightness_p))

    if p.rgb_shift > 0:
        transforms.append(A.RGBShift(
            r_shift_limit=p.rgb_shift, g_shift_limit=p.rgb_shift,
            b_shift_limit=p.rgb_shift, p=p.rgb_p))

    local_group = []
    if p.clahe:
        local_group.append(A.CLAHE(clip_limit=3.0, p=1.0))
    if p.equalize:
        local_group.append(A.Equalize(p=1.0))
    if local_group:
        transforms.append(A.OneOf(local_group, p=p.clahe_p))

    if p.gauss_noise > 0:
        transforms.append(A.GaussNoise(std_range=(0.01, p.gauss_noise), p=p.gauss_noise_p))

    blur_group = []
    if p.gaussian_blur:
        blur_group.append(A.GaussianBlur(blur_limit=(3, 7), p=1.0))
    if p.motion_blur:
        blur_group.append(A.MotionBlur(blur_limit=(3, 7), p=1.0))
    if blur_group:
        transforms.append(A.OneOf(blur_group, p=p.motion_blur_p))

    if p.cutout:
        transforms.append(A.CoarseDropout(
            num_holes_range=(1, 4), hole_height_range=(0.05, 0.15),
            hole_width_range=(0.05, 0.15), p=p.cutout_p))

    if has_polygons:
        return A.Compose(transforms,
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))
    return A.Compose(transforms)


def run_augmentation(img: np.ndarray, params: AugParams, polygons: list) -> dict:
    has_polygons = len(polygons) > 0
    transform = build_transform(params, has_polygons)

    if has_polygons:
        h, w = img.shape[:2]
        kps_norm, grain_sizes = polygons_to_keypoints(polygons)
        kps_px = [(x * w, y * h) for x, y in kps_norm]
        result = transform(image=img, keypoints=kps_px)
        aug_img = result["image"]
        aug_h, aug_w = aug_img.shape[:2]
        norm_kps = [
            (min(max(float(k[0]) / aug_w, 0.0), 1.0),
             min(max(float(k[1]) / aug_h, 0.0), 1.0))
            if k is not None else None
            for k in result["keypoints"]
        ]
        aug_polygons = keypoints_to_polygons(norm_kps, grain_sizes)
    else:
        result = transform(image=img)
        aug_img = result["image"]
        aug_polygons = []

    return {"image": aug_img, "polygons": aug_polygons}


# ─────────────────────────────────────────────────────────────
#  Drawing
# ─────────────────────────────────────────────────────────────

def draw_polygons(img: np.ndarray, polygons: list) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    WHITE = (255, 255, 255)

    # Spread hues evenly so adjacent polygons are visually distinct
    n = max(len(polygons), 1)
    for idx, poly in enumerate(polygons):
        if len(poly) < 3:
            continue
        hue = int(idx * 180 / n)          # OpenCV hue: 0–179
        hsv_fill = np.uint8([[[hue, 200, 200]]])
        bgr = cv2.cvtColor(hsv_fill, cv2.COLOR_HSV2BGR)[0][0]
        color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))

        pts = np.array([[int(x * w), int(y * h)] for x, y in poly], dtype=np.int32)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.15, out, 0.85, 0, out)
        cv2.polylines(out, [pts], isClosed=True, color=WHITE, thickness=4)
        cv2.polylines(out, [pts], isClosed=True, color=color,  thickness=2)
        for pt in pts:
            cv2.circle(out, tuple(pt), 5, WHITE, -1)
            cv2.circle(out, tuple(pt), 3, color,  -1)
    return out


# ─────────────────────────────────────────────────────────────
#  Image helpers
# ─────────────────────────────────────────────────────────────

def b64_to_numpy(b64: str) -> np.ndarray:
    data = base64.b64decode(b64.split(",")[-1])
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


def numpy_to_b64(img: np.ndarray, quality: int = 88) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("encode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def polygons_to_yolo_bboxes(polygons: list) -> list:
    bboxes = []
    for poly in polygons:
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        x1, x2 = min(xs), max(xs); y1, y2 = min(ys), max(ys)
        bboxes.append([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1])
    return bboxes


# ─────────────────────────────────────────────────────────────
#  Code generation
# ─────────────────────────────────────────────────────────────

def build_code(p: AugParams) -> str:
    lines = ["import albumentations as A", "", "transform = A.Compose(["]
    if p.hflip:   lines.append("    A.HorizontalFlip(p=0.5),")
    if p.vflip:   lines.append("    A.VerticalFlip(p=0.5),")
    if p.rot90:   lines.append("    A.RandomRotate90(p=0.5),")
    if p.rotate_limit > 0:
        lines.append(f"    A.Rotate(limit={p.rotate_limit}, p={p.rotate_p}),")
    if p.scale_min != 1.0 or p.scale_max != 1.0:
        lines.append(f"    A.RandomScale(scale_limit=({p.scale_min-1.0:.1f}, {p.scale_max-1.0:.1f}), p={p.scale_p}),")
    cp = []
    if p.brightness_limit > 0:
        cp.append(f"        A.RandomBrightnessContrast(brightness_limit={p.brightness_limit}, contrast_limit={p.contrast_limit}, p=1.0),")
    if p.hue_shift > 0:
        cp.append(f"        A.HueSaturationValue(hue_shift_limit={p.hue_shift}, sat_shift_limit={p.sat_shift}, val_shift_limit={p.val_shift}, p=1.0),")
    if cp:
        lines += [f"    A.OneOf(["] + cp + [f"    ], p={p.brightness_p}),"]
    if p.rgb_shift > 0:
        lines.append(f"    A.RGBShift(r_shift_limit={p.rgb_shift}, g_shift_limit={p.rgb_shift}, b_shift_limit={p.rgb_shift}, p={p.rgb_p}),")
    lp = []
    if p.clahe:    lp.append("        A.CLAHE(clip_limit=3.0, p=1.0),")
    if p.equalize: lp.append("        A.Equalize(p=1.0),")
    if lp:
        lines += [f"    A.OneOf(["] + lp + [f"    ], p={p.clahe_p}),"]
    if p.gauss_noise > 0:
        lines.append(f"    A.GaussNoise(std_range=(0.01, {p.gauss_noise}), p={p.gauss_noise_p}),")
    bp = []
    if p.gaussian_blur: bp.append("        A.GaussianBlur(blur_limit=(3, 7), p=1.0),")
    if p.motion_blur:   bp.append("        A.MotionBlur(blur_limit=(3, 7), p=1.0),")
    if bp:
        lines += [f"    A.OneOf(["] + bp + [f"    ], p={p.motion_blur_p}),"]
    if p.cutout:
        lines.append(f"    A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.05, 0.15), hole_width_range=(0.05, 0.15), p={p.cutout_p}),")
    lines.append("])")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html") as f:
        return f.read()


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    return JSONResponse({"image_b64": numpy_to_b64(img, quality=92)})


@app.post("/preview_labels")
async def preview_labels(req: AugRequest):
    """Return original image twice: clean and with polygons drawn."""
    try:
        img = b64_to_numpy(req.image_b64)
    except Exception as e:
        raise HTTPException(400, f"Could not decode image: {e}")
    polygons = parse_polygon_labels(req.label_text) if req.label_text.strip() else []
    clean_b64  = numpy_to_b64(img)
    annot_b64  = numpy_to_b64(draw_polygons(img, polygons)) if polygons else clean_b64
    return JSONResponse({
        "clean_b64":  clean_b64,
        "annot_b64":  annot_b64,
        "grain_count": len(polygons),
    })


@app.post("/augment")
async def augment(req: AugRequest):
    try:
        img = b64_to_numpy(req.image_b64)
    except Exception as e:
        raise HTTPException(400, f"Could not decode image: {e}")

    polygons = parse_polygon_labels(req.label_text) if req.label_text.strip() else []
    random.seed(req.params.seed)
    np.random.seed(req.params.seed)
    out = run_augmentation(img, req.params, polygons)

    # Always render BOTH clean and annotated versions with the SAME augmented pixels
    aug_clean = numpy_to_b64(out["image"])
    aug_annot = numpy_to_b64(draw_polygons(out["image"], out["polygons"])) if out["polygons"] else aug_clean
    orig_clean = numpy_to_b64(img)
    orig_annot = numpy_to_b64(draw_polygons(img, polygons)) if polygons else orig_clean

    return JSONResponse({
        "orig_clean_b64":     orig_clean,
        "orig_annot_b64":     orig_annot,
        "aug_clean_b64":      aug_clean,
        "aug_annot_b64":      aug_annot,
        "aug_polygon_count":  len(out["polygons"]),
        "code":               build_code(req.params),
    })


@app.post("/augment_grid")
async def augment_grid(req: AugRequest):
    try:
        img = b64_to_numpy(req.image_b64)
    except Exception as e:
        raise HTTPException(400, f"Could not decode image: {e}")

    polygons = parse_polygon_labels(req.label_text) if req.label_text.strip() else []
    n = min(req.params.grid_n, 12)
    clean_grid = []
    annot_grid = []

    for i in range(n):
        random.seed(req.params.seed + i)
        np.random.seed(req.params.seed + i)
        out = run_augmentation(img, req.params, polygons)
        clean_b64 = numpy_to_b64(out["image"], quality=80)
        annot_b64 = numpy_to_b64(draw_polygons(out["image"], out["polygons"]), quality=80) \
                    if out["polygons"] else clean_b64
        clean_grid.append(clean_b64)
        annot_grid.append(annot_b64)

    return JSONResponse({"clean_grid": clean_grid, "annot_grid": annot_grid})


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)