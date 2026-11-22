"""Geometry-preserving, deterministic image perturbations."""

import cv2
import numpy as np


def transform(image: np.ndarray, name: str, severity: int, seed: int) -> tuple[np.ndarray, dict]:
    if severity not in range(1, 6):
        raise ValueError("Severity must be 1–5")
    rng = np.random.default_rng(seed)
    policy = "geometry-preserving"
    if name == "brightness":
        alpha, beta = 1 - 0.12 * severity, -3 * severity
        result = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        params = {"alpha": alpha, "beta": beta}
    elif name == "brightness_contrast":
        alpha, beta = 1 + 0.10 * severity, -8 * severity
        result = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        params = {"alpha": alpha, "beta": beta}
    elif name == "gaussian_blur":
        kernel = 2 * severity + 1
        result = cv2.GaussianBlur(image, (kernel, kernel), 0)
        params = {"kernel": kernel}
    elif name == "motion_blur":
        kernel = 2 * severity + 3
        angle = int(rng.choice([0, 45, 90, 135]))
        matrix = np.zeros((kernel, kernel), dtype=np.float32)
        center = kernel // 2
        directions = {0: ((0, center), (kernel - 1, center)), 45: ((0, 0), (kernel - 1, kernel - 1)), 90: ((center, 0), (center, kernel - 1)), 135: ((kernel - 1, 0), (0, kernel - 1))}
        cv2.line(matrix, *directions[angle], 1, 1)
        matrix /= matrix.sum()
        result = cv2.filter2D(image, -1, matrix)
        params = {"kernel": kernel, "angle_degrees": angle}
    elif name == "gaussian_noise":
        sigma = 5 * severity
        noise = rng.normal(0, sigma, image.shape)
        result = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        params = {"sigma": sigma, "distribution": "normal(0,sigma)", "generator": "numpy.default_rng"}
    elif name == "jpeg":
        quality = 95 - severity * 15
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise ValueError("JPEG encoding failed")
        result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        params = {"quality": quality}
    elif name == "fog":
        alpha = 0.08 * severity
        height, width = image.shape[:2]
        haze = rng.normal(0, 20, (height, width)).astype(np.float32)
        haze = cv2.GaussianBlur(haze, (0, 0), sigmaX=max(height, width) / 12)
        veil = np.clip(220 + haze, 0, 255)[:, :, None]
        result = np.clip(image.astype(np.float32) * (1 - alpha) + veil * alpha, 0, 255).astype(np.uint8)
        params = {"alpha": alpha, "veil_mean": 220, "veil_noise_sigma": 20, "blur_scale": 12}
    elif name == "rain":
        height, width = image.shape[:2]
        count = 12 * severity
        length = 6 + 3 * severity
        opacity = min(0.15 + 0.04 * severity, 0.5)
        overlay = image.copy()
        strokes = []
        for _ in range(count):
            x = int(rng.integers(0, width))
            y = int(rng.integers(0, height))
            end = (min(width - 1, x + max(1, length // 4)), min(height - 1, y + length))
            cv2.line(overlay, (x, y), end, (205, 215, 225), 1)
            strokes.append([x, y, end[0], end[1]])
        result = cv2.addWeighted(image, 1 - opacity, overlay, opacity, 0)
        params = {"count": count, "length": length, "opacity": opacity, "strokes": strokes}
    elif name == "partial_occlusion":
        height, width = image.shape[:2]
        fraction = 0.06 + 0.05 * severity
        box_width = max(1, round(width * fraction))
        box_height = max(1, round(height * fraction))
        x = int(rng.integers(0, max(1, width - box_width + 1)))
        y = int(rng.integers(0, max(1, height - box_height + 1)))
        result = image.copy()
        result[y:y + box_height, x:x + box_width] = 0
        params = {"rectangle_xywh": [x, y, box_width, box_height], "fill_bgr": [0, 0, 0]}
        policy = "exploratory: occlusion may hide labeled objects; metrics are not core label-valid results"
    elif name == "color_shift":
        hue_shift = 3 * severity
        saturation_scale = 1 - 0.08 * severity
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        params = {"hue_shift_opencv_units": hue_shift, "saturation_scale": saturation_scale}
    elif name == "defocus":
        radius = severity + 1
        coordinates = np.arange(-radius, radius + 1)
        xx, yy = np.meshgrid(coordinates, coordinates)
        disk = (xx * xx + yy * yy <= radius * radius).astype(np.float32)
        disk /= disk.sum()
        result = cv2.filter2D(image, -1, disk)
        params = {"disk_radius": radius, "kernel_size": 2 * radius + 1}
    else:
        raise ValueError(f"Unknown stress condition: {name}")
    return result, {"name": name, "severity": severity, "seed": seed, "parameters": params, "label_policy": policy}
