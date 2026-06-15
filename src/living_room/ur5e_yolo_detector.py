from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import cv2


@dataclass
class YoloMaskDetection:
    cls_id: int
    cls_name: str
    conf: float
    bbox_xyxy: Tuple[int, int, int, int]
    center_px: Tuple[float, float]
    area_px: int
    mask: np.ndarray
    yaw_deg: float
    major_len_px: float
    minor_len_px: float


class YoloMaskDetector:
    def __init__(self, model_path: str = "yolo11n-seg.pt", conf: float = 0.20):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = conf
        self.names = self.model.names

    def detect(self, image_rgb: np.ndarray) -> List[YoloMaskDetection]:
        # Ultralytics live ndarray color convention can differ from saved jpg path.
        # Try RGB first. If no masks, try BGR fallback.
        results = self.model.predict(image_rgb, conf=self.conf, verbose=False)

        if results:
            r = results[0]
            if r.boxes is not None and r.masks is not None and len(r.boxes) > 0:
                pass
            else:
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                results = self.model.predict(image_bgr, conf=self.conf, verbose=False)
        else:
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            results = self.model.predict(image_bgr, conf=self.conf, verbose=False)

        if not results:
            return []

        r = results[0]
        if r.boxes is None or r.masks is None:
            return []

        masks = r.masks.data.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()

        h, w = image_rgb.shape[:2]
        detections = []

        for mask_f, box, cls_id, cf in zip(masks, boxes, cls_ids, confs):
            mask = (mask_f > 0.5).astype(np.uint8)
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            area = int(mask.sum())
            if area < 20:
                continue

            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue

            cx = float(xs.mean())
            cy = float(ys.mean())

            pts = np.stack([xs, ys], axis=1).astype(np.float32)
            yaw_deg = 0.0
            major_len = 0.0
            minor_len = 0.0

            if len(pts) >= 5:
                mean = pts.mean(axis=0)
                cov = np.cov((pts - mean).T)
                vals, vecs = np.linalg.eigh(cov)
                order = np.argsort(vals)[::-1]
                vals = vals[order]
                vec = vecs[:, order[0]]
                yaw_deg = float(np.degrees(np.arctan2(vec[1], vec[0])))
                major_len = float(4.0 * np.sqrt(max(vals[0], 0.0)))
                minor_len = float(4.0 * np.sqrt(max(vals[1], 0.0)))

            x1, y1, x2, y2 = [int(v) for v in box]
            name = str(self.names.get(int(cls_id), str(cls_id)))

            detections.append(
                YoloMaskDetection(
                    cls_id=int(cls_id),
                    cls_name=name,
                    conf=float(cf),
                    bbox_xyxy=(x1, y1, x2, y2),
                    center_px=(cx, cy),
                    area_px=area,
                    mask=mask,
                    yaw_deg=yaw_deg,
                    major_len_px=major_len,
                    minor_len_px=minor_len,
                )
            )

        detections.sort(key=lambda d: d.conf, reverse=True)
        return detections


def draw_yolo_detections(image_rgb: np.ndarray, detections: List[YoloMaskDetection]) -> np.ndarray:
    vis = image_rgb.copy()

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox_xyxy
        cx, cy = det.center_px

        color = (0, 255, 0)

        overlay = vis.copy()
        overlay[det.mask > 0] = (0.5 * overlay[det.mask > 0] + 0.5 * np.array(color)).astype(np.uint8)
        vis = overlay

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.circle(vis, (int(cx), int(cy)), 4, (255, 0, 0), -1)

        label = f"{det.cls_name} {det.conf:.2f} area={det.area_px} yaw={det.yaw_deg:.1f}"
        cv2.putText(
            vis,
            label,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    return vis
