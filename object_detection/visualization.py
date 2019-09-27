from os import PathLike

import numpy as np
import cv2
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

from .output import DetectionOutput


def visualize(output: DetectionOutput, image_path: PathLike):
    for detection in output.detections:
        if detection.confidence < 0.5:
            break

        image = read_image_bgr(output.image_name)
        canvas = image.copy()
        # canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        box = np.array(
            [detection.bbox.x0, detection.bbox.x1, detection.bbox.y0, detection.bbox.y1]
        ).astype(int)

        draw_box(canvas, box, color=(0, 255, 0))
        caption = f"{detection.label} {detection.confidence:.3f}"
        draw_caption(canvas, box, caption)
        cv2.imwrite(image_path, canvas)
