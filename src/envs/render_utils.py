import cv2
import numpy as np


def rgb_to_bgr(img, flip=True):
    if img is None:
        return None
    if flip:
        img = img[::-1]
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def make_camera_panel(obs, camera_keys, title_text=None, scale=3):
    frames = []

    for key in camera_keys:
        if key not in obs:
            continue

        frame = rgb_to_bgr(obs[key], flip=True)
        label = key.replace("_image", "")

        cv2.putText(
            frame,
            label,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        frames.append(frame)

    if len(frames) == 0:
        return None

    h = min(f.shape[0] for f in frames)
    resized = []

    for f in frames:
        ratio = h / f.shape[0]
        w = int(f.shape[1] * ratio)
        resized.append(cv2.resize(f, (w, h)))

    panel = np.concatenate(resized, axis=1)

    if title_text is not None:
        cv2.putText(
            panel,
            title_text,
            (8, panel.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if scale != 1:
        panel = cv2.resize(
            panel,
            (panel.shape[1] * scale, panel.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )

    return panel


def show_camera_panel(
    obs,
    camera_keys,
    window_name="Mini RT-2 Camera Views",
    title_text=None,
    delay=1,
):
    panel = make_camera_panel(
        obs=obs,
        camera_keys=camera_keys,
        title_text=title_text,
    )

    if panel is None:
        return False

    cv2.imshow(window_name, panel)
    key = cv2.waitKey(delay) & 0xFF

    if key == ord("q") or key == 27:
        return True

    return False


def close_render_windows():
    cv2.destroyAllWindows()
