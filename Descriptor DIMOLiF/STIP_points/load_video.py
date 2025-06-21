import cv2
import numpy as np

def load_video_to_array(path, max_frames=16000, resize_factor=1.0):
    """
    Carga un video y lo convierte a un arreglo 3D (H, W, T) en escala de grises.

    Args:
        path (str): Ruta al archivo de video.
        max_frames (int or None): Número máximo de frames a leer. Si None, se leen todos.
        resize_factor (float): Factor de escala para reducir resolución (1.0 = original).

    Returns:
        np.array: arreglo (alto, ancho, tiempo) normalizado a [0, 1].
    """
    cap = cv2.VideoCapture(path)
    frames = []
    count = 0

    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el video: {path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (max_frames is not None and count >= max_frames):
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if resize_factor != 1.0:
            gray = cv2.resize(gray, (0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)

        frames.append(gray.astype(np.float32) / 255.0)
        count += 1

    cap.release()

    if not frames:
        raise ValueError(f"No se pudieron leer frames de {path}")

    print(f"{path} | Frames leídos: {count} | Shape final: {frames[0].shape} x {count}")
    video = np.stack(frames, axis=-1)
    return video
