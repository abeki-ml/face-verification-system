import cv2
import numpy as np
from insightface.app import FaceAnalysis

model = FaceAnalysis(name="buffalo_sc", allowed_modules=["detection", "recognition"])
model.prepare(ctx_id=-1)


def read_image(file_bytes):
    numbers = np.frombuffer(file_bytes, dtype=np.uint8)
    return cv2.imdecode(numbers, cv2.IMREAD_COLOR)


def get_fingerprint(image):
    faces = model.get(image)
    if len(faces) == 0:
        return None
    return faces[0].normed_embedding