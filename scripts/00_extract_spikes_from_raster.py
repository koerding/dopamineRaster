#!/usr/bin/env python3
"""
Approximate spike extraction from a raster scan.

This script recreates the workflow used in the chat:
1. detect the three raster panels
2. estimate event line locations
3. suppress line pixels
4. detect circular/dot-like spike marks, including candidates touching lines
5. merge near-duplicate detections
6. convert x positions to seconds using panel-specific calibration
7. save spike times and diagnostic images

This is intentionally practical and easy to modify, not a perfect OCR system.
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "dopamine_raster_source.png"
DATA = ROOT / "data"
FIGS = ROOT / "figures"
NOTES = ROOT / "notes"

DATA.mkdir(exist_ok=True, parents=True)
FIGS.mkdir(exist_ok=True, parents=True)
NOTES.mkdir(exist_ok=True, parents=True)

PANEL_ORDER = [
    "no_prediction_reward_occurs",
    "reward_predicted_reward_occurs",
    "reward_predicted_no_reward_occurs",
]


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def detect_panels(gray: np.ndarray):
    """
    Detect three large horizontal raster regions by thresholding dark content
    and finding wide connected components.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    bw = (blur < 210).astype(np.uint8) * 255

    # strengthen long horizontal structure
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 5))
    merged = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    candidates = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if w > gray.shape[1] * 0.45 and h > 60 and area > 8000:
            candidates.append((x, y, w, h, area))

    # sort by vertical position and keep the three largest plausible components
    candidates = sorted(candidates, key=lambda t: t[1])
    if len(candidates) < 3:
        raise RuntimeError("Could not detect three raster panels")

    # keep three best by height*width while preserving order
    top3 = sorted(sorted(candidates, key=lambda t: t[2] * t[3], reverse=True)[:3], key=lambda t: t[1])
    panels = []
    for name, (x, y, w, h, _) in zip(PANEL_ORDER, top3):
        pad_x = 8
        pad_y = 8
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(gray.shape[1], x + w + pad_x)
        y1 = min(gray.shape[0], y + h + pad_y)
        panels.append(dict(panel=name, x0=x0, y0=y0, x1=x1, y1=y1, width=x1 - x0, height=y1 - y0))
    return panels


def estimate_event_lines(panel_img: np.ndarray):
    """
    Estimate strong vertical event lines from the column darkness profile.
    Returns a list of x positions.
    """
    col_dark = (255 - panel_img).mean(axis=0)
    smooth = cv2.GaussianBlur(col_dark.reshape(1, -1).astype(np.float32), (1, 1), 0).ravel()
    thresh = np.percentile(smooth, 97)
    idx = np.where(smooth >= thresh)[0]
    if len(idx) == 0:
        return []

    groups = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i <= prev + 3:
            prev = i
        else:
            groups.append((start, prev))
            start = prev = i
    groups.append((start, prev))
    centers = [int((a + b) / 2) for a, b in groups if (b - a + 1) >= 1]
    return centers


def suppress_lines(panel_img: np.ndarray, line_xs, radius=3):
    """
    Paint around event lines with local background median so line-touching dots
    can later be reintroduced from residual darkness.
    """
    img = panel_img.copy()
    h, w = img.shape
    for x in line_xs:
        x0 = max(0, x - radius)
        x1 = min(w, x + radius + 1)
        left = max(0, x0 - 12)
        right = min(w, x1 + 12)
        background = np.hstack([img[:, left:x0], img[:, x1:right]]) if (x0 > left or right > x1) else img
        fill = int(np.median(background))
        img[:, x0:x1] = fill
    return img


def blob_candidates(img: np.ndarray):
    """
    Detect dark circular-ish components after adaptive thresholding.
    """
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    bw = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7
    )

    # remove tiny specks, keep compact blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    detections = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        if area < 3 or area > 150:
            continue
        aspect = w / max(h, 1)
        if aspect < 0.35 or aspect > 2.8:
            continue
        # circular-ish score from fill ratio
        box_area = max(w * h, 1)
        fill_ratio = area / box_area
        if fill_ratio < 0.12 or fill_ratio > 1.0:
            continue
        sigma = float(max(w, h) / 2.5)
        score = float(area * fill_ratio)
        detections.append(dict(cx=float(cx), cy=float(cy), sigma_px=sigma, score=score))
    return detections, bw


def add_line_touching_residuals(original: np.ndarray, suppressed: np.ndarray, line_xs):
    """
    Look for residual darkness near event lines that may correspond to dots
    touching or overlapping the lines.
    """
    residual = cv2.subtract(
        (255 - suppressed).astype(np.uint8),
        cv2.GaussianBlur((255 - suppressed).astype(np.uint8), (0, 0), 1.2),
    )
    dark = (255 - original).astype(np.int16) - (255 - suppressed).astype(np.int16)
    band = np.zeros_like(original, dtype=np.uint8)
    for x in line_xs:
        x0 = max(0, x - 7)
        x1 = min(original.shape[1], x + 8)
        band[:, x0:x1] = 255

    target = np.where((dark > 10) & (band > 0), 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    target = cv2.morphologyEx(target, cv2.MORPH_OPEN, kernel, iterations=1)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(target, connectivity=8)
    detections = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        if area < 2 or area > 120:
            continue
        aspect = w / max(h, 1)
        if aspect < 0.25 or aspect > 4.0:
            continue
        sigma = float(max(w, h) / 2.5)
        score = float(area)
        detections.append(dict(cx=float(cx), cy=float(cy), sigma_px=sigma, score=score))
    return detections, target


def merge_detections(detections, dist_thresh=4.0):
    """
    Greedy merge of nearby detections. Useful when touching dots get split.
    """
    if not detections:
        return []

    pts = [d.copy() for d in sorted(detections, key=lambda d: d["score"], reverse=True)]
    kept = []
    for d in pts:
        merged = False
        for k in kept:
            dx = d["cx"] - k["cx"]
            dy = d["cy"] - k["cy"]
            if (dx * dx + dy * dy) ** 0.5 <= dist_thresh:
                total = k["score"] + d["score"]
                k["cx"] = (k["cx"] * k["score"] + d["cx"] * d["score"]) / total
                k["cy"] = (k["cy"] * k["score"] + d["cy"] * d["score"]) / total
                k["sigma_px"] = max(k["sigma_px"], d["sigma_px"])
                k["score"] = total
                merged = True
                break
        if not merged:
            kept.append(d)
    return kept


def calibrate_time(panel_name: str, x: float, width: int) -> float:
    """
    Approximate x-to-time map used for the extracted results.
    All panels end at 2 s. Panel 1 starts at -0.5 s.
    Panels 2 and 3 start at -1 s.
    """
    if panel_name == "no_prediction_reward_occurs":
        t0, t1 = -0.5, 2.0
    else:
        t0, t1 = -1.0, 2.0
    return t0 + (x / max(width - 1, 1)) * (t1 - t0)


def draw_overlay(panel_img, detections, line_xs):
    rgb = cv2.cvtColor(panel_img, cv2.COLOR_GRAY2BGR)
    for x in line_xs:
        cv2.line(rgb, (int(x), 0), (int(x), panel_img.shape[0] - 1), (40, 160, 240), 1)
    for d in detections:
        center = (int(round(d["cx"])), int(round(d["cy"])))
        rad = max(2, int(round(d["sigma_px"])))
        cv2.circle(rgb, center, rad, (240, 80, 80), 1)
        cv2.circle(rgb, center, 1, (250, 250, 250), -1)
    return rgb


def draw_reconstruction(shape, detections, line_xs):
    canvas = np.full(shape, 255, dtype=np.uint8)
    for x in line_xs:
        cv2.line(canvas, (int(x), 0), (int(x), shape[0] - 1), 130, 1)
    for d in detections:
        center = (int(round(d["cx"])), int(round(d["cy"])))
        rad = max(1, int(round(d["sigma_px"])))
        cv2.circle(canvas, center, rad, 0, -1)
    return canvas


def main():
    gray = read_gray(RAW)
    panels = detect_panels(gray)

    rows = []
    panel_meta = []
    overlay_rows = []
    recon_rows = []

    for p in panels:
        crop = gray[p["y0"]:p["y1"], p["x0"]:p["x1"]]
        line_xs = estimate_event_lines(crop)
        suppressed = suppress_lines(crop, line_xs, radius=3)

        det_a, _ = blob_candidates(suppressed)
        det_b, _ = add_line_touching_residuals(crop, suppressed, line_xs)
        merged = merge_detections(det_a + det_b, dist_thresh=4.0)

        for j, d in enumerate(merged, start=1):
            rows.append({
                "panel": p["panel"],
                "spike_id": j,
                "time_s": calibrate_time(p["panel"], d["cx"], p["width"]),
                "x_px_in_panel": d["cx"],
                "y_px_in_panel": d["cy"],
                "x_px_abs": p["x0"] + d["cx"],
                "y_px_abs": p["y0"] + d["cy"],
                "y_norm_in_raster": d["cy"] / max(p["height"] - 1, 1),
                "sigma_px": d["sigma_px"],
                "detection_score": d["score"],
            })

        panel_meta.append({
            "panel": p["panel"],
            "x0": p["x0"], "y0": p["y0"], "x1": p["x1"], "y1": p["y1"],
            "width": p["width"], "height": p["height"],
            "n_detected_spikes": len(merged),
            "estimated_event_lines_x": ",".join(map(str, line_xs)),
        })

        overlay_rows.append(draw_overlay(crop, merged, line_xs))
        recon_rows.append(draw_reconstruction(crop.shape, merged, line_xs))

    spikes = pd.DataFrame(rows)
    meta = pd.DataFrame(panel_meta)

    spikes.to_csv(DATA / "dopamine_raster_spike_times.csv", index=False)
    meta.to_csv(DATA / "dopamine_raster_panel_metadata.csv", index=False)

    if overlay_rows:
        overlay = cv2.vconcat(overlay_rows)
        cv2.imwrite(str(FIGS / "dopamine_raster_detection_overlay.png"), overlay)
    if recon_rows:
        recon = cv2.vconcat(recon_rows)
        cv2.imwrite(str(FIGS / "dopamine_raster_reconstruction.png"), cv2.vconcat(recon_rows))

    note = """\
Approximate spike extraction from a low-resolution raster image.

Key ideas:
- treat the image as the sum of vertical event lines and circular spike dots
- suppress lines, then seed candidate dots from residual darkness
- merge nearby detections to handle touching dots
- convert x coordinate to time using panel-specific axis assumptions

This is approximate digitization, not exact ground truth.
"""
    (NOTES / "dopamine_raster_extraction_notes.txt").write_text(note)


if __name__ == "__main__":
    main()
