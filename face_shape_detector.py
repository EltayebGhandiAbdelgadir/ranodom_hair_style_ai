import cv2
import mediapipe as mp
import numpy as np

def detect_face_shape(image_path):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    
    img = cv2.imread(image_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return "Unknown"

    h, w, _ = img.shape
    landmarks = results.multi_face_landmarks[0].landmark
    coords = np.array([[int(p.x * w), int(p.y * h)] for p in landmarks])

    jaw_width = np.linalg.norm(coords[234] - coords[454])  # jawline
    face_length = np.linalg.norm(coords[10] - coords[152])  # chin to forehead
    cheekbone_width = np.linalg.norm(coords[93] - coords[323])
    
    ratio = jaw_width / face_length

    if ratio < 0.8:
        return "Oval"
    elif ratio < 1.0:
        return "Round"
    elif cheekbone_width > jaw_width:
        return "Heart"
    else:
        return "Square"
