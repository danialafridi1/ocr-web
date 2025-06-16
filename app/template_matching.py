# template_matching_module.py
"""
Module providing preprocessing and template matching functions:
- Noise Reduction
- Binarization
- Skew Correction
- Normalization
- Template Matching
"""
import cv2
import numpy as np
import os


def preprocess_image_cv(img_cv):
    """
    Apply preprocessing on a CV image array:
      1. Noise Reduction (Gaussian Blur)
      2. Binarization (Otsu)
      3. Skew Correction
      4. Normalization (fixed height)
    """
    # 1. Noise Reduction
    denoised = cv2.GaussianBlur(img_cv, (5, 5), 0)

    # 2. Binarization
    _, binary = cv2.threshold(
        denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Skew Correction
    coords = np.column_stack(np.where(binary < 255))
    if coords.shape[0] > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = binary.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(
            binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    else:
        deskewed = binary

    # 4. Normalization
    norm_height = 100
    scale = norm_height / float(deskewed.shape[0])
    norm_width = int(deskewed.shape[1] * scale)
    normalized = cv2.resize(
        deskewed, (norm_width, norm_height), interpolation=cv2.INTER_AREA)

    return normalized


def match_template(img_cv, template_path, threshold=0.8):
    """
    Preprocess input image and template, perform template matching.
    Returns list of bounding boxes where template is found.
    """
    # Preprocess input
    pre_img = preprocess_image_cv(img_cv)

    # Load and preprocess template
    tmpl = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if tmpl is None:
        raise FileNotFoundError(f"Template not found: {template_path}")
    _, tmpl_bin = cv2.threshold(
        tmpl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    th, tw = tmpl_bin.shape

    # Match
    res = cv2.matchTemplate(pre_img, tmpl_bin, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    boxes = []
    for pt in zip(*loc[::-1]):
        boxes.append((pt[0], pt[1], tw, th, float(res[pt[1], pt[0]])))
    return boxes
