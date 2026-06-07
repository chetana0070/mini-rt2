import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import cv2


@dataclass
class MaskFeatures:
    seg_id: int
    geom_name: str
    area_px: int
    bbox_xywh: Tuple[int, int, int, int]
    center_px: Tuple[float, float]
    aspect_ratio: float
    major_axis_deg: float


def normalize_seg(seg):
    seg = np.asarray(seg)
    if seg.ndim == 3:
        seg = seg[..., 0]
    return seg.astype(np.int32)


def geom_name(sim, seg_id):
    try:
        name = sim.model.geom_id2name(int(seg_id))
        return str(name) if name is not None else f"seg_{seg_id}"
    except Exception:
        return f"seg_{seg_id}"


def one_feature(seg, seg_id, name):
    mask = seg == seg_id
    ys, xs = np.where(mask)

    if len(xs) < 80:
        return None

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    w = max(1, x1 - x0 + 1)
    h = max(1, y1 - y0 + 1)

    cx = float(xs.mean())
    cy = float(ys.mean())

    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    coords = coords - coords.mean(axis=0, keepdims=True)

    yaw_deg = 0.0
    if len(coords) >= 3:
        cov = np.cov(coords, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major = eigvecs[:, np.argmax(eigvals)]
        yaw_deg = math.degrees(math.atan2(float(major[1]), float(major[0])))

    aspect = float(max(w, h)) / float(max(1, min(w, h)))

    return MaskFeatures(
        seg_id=int(seg_id),
        geom_name=name,
        area_px=int(len(xs)),
        bbox_xywh=(x0, y0, w, h),
        center_px=(cx, cy),
        aspect_ratio=aspect,
        major_axis_deg=yaw_deg,
    )


def extract_mask_features(seg, sim=None, min_area=80):
    seg = normalize_seg(seg)
    feats = []

    for sid in np.unique(seg):
        sid = int(sid)

        if sid in [-1, 0]:
            continue

        if int((seg == sid).sum()) < min_area:
            continue

        feat = one_feature(seg, sid, geom_name(sim, sid))
        if feat is not None:
            feats.append(feat)

    feats.sort(key=lambda f: f.area_px, reverse=True)
    return feats


def select_object_feature(features: List[MaskFeatures], object_name: str) -> Optional[MaskFeatures]:
    if not features:
        return None

    aliases = {
        "mug": ["mug", "milk"],
        "milk": ["mug", "milk"],
        "book": ["book", "bread"],
        "bread": ["book", "bread"],
        "remote": ["remote", "can"],
        "can": ["remote", "can"],
        "toy": ["toy", "cereal", "box"],
        "cereal": ["toy", "cereal", "box"],
    }.get(object_name.lower(), [object_name.lower()])

    matches = []
    for f in features:
        low = f.geom_name.lower()
        if any(a in low for a in aliases):
            matches.append(f)

    if matches:
        matches.sort(key=lambda f: f.area_px, reverse=True)
        return matches[0]

    # Important: do NOT fallback to biggest mask.
    # Biggest mask is usually wall / table / robot.
    return None


def shape_hint(object_name, feat):
    name = object_name.lower()

    if name in ["mug", "milk"]:
        return "compact", 0.035, False
    if name in ["book", "bread"]:
        return "flat_rectangular", 0.030, False
    if name in ["remote", "can"]:
        return "long_thin", 0.030, False
    if name in ["toy", "cereal"]:
        return "box_rectangular", 0.045, True

    if feat.aspect_ratio > 1.8:
        return "long_rectangular", 0.035, True

    return "compact_unknown", 0.035, False


def draw_features(img, features, selected=None):
    vis = img.copy()
    if vis.dtype != np.uint8:
        vis = np.clip(vis, 0, 255).astype(np.uint8)

    for f in features:
        x, y, w, h = f.bbox_xywh
        cx, cy = f.center_px

        thick = 3 if selected is not None and selected.seg_id == f.seg_id else 1

        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 255), thick)
        cv2.circle(vis, (int(cx), int(cy)), 4, (255, 255, 255), -1)

        label = f"{f.seg_id}:{f.geom_name[:28]} ar={f.aspect_ratio:.2f}"
        cv2.putText(
            vis,
            label,
            (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return vis
