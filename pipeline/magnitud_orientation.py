import numpy as np
import cv2

def compute_mag_ori_from_stc(stc):
    """
    Calcula la magnitud y orientación del flujo óptico a partir de un STC.

    Args:
        stc (np.array): volumen (10, 10, 5), escala [0, 1] o [0, 255]

    Returns:
        magnitudes: np.array de tamaño (10, 10, 4)
        orientations: np.array de tamaño (10, 10, 4) en radianes
    """
    h, w, t = stc.shape
    assert t >= 2, "Se requieren al menos 2 frames para flujo óptico"

    # Si viene en float32 [0, 1], convertir a uint8
    if stc.dtype != np.uint8:
        stc = (stc * 255).astype(np.uint8)

    magnitudes = np.zeros((h, w, t - 1), dtype=np.float32)
    orientations = np.zeros((h, w, t - 1), dtype=np.float32)

    for i in range(t - 1):
        prev = stc[:, :, i]
        next = stc[:, :, i + 1]

        flow = cv2.calcOpticalFlowFarneback(prev, next, None,
                                            pyr_scale=0.5, levels=1, winsize=5,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        flow_x, flow_y = flow[..., 0], flow[..., 1]
        mag = np.sqrt(flow_x**2 + flow_y**2)
        ori = np.arctan2(flow_y, flow_x)  # orientación en radianes

        magnitudes[:, :, i] = mag
        orientations[:, :, i] = ori

    return magnitudes, orientations
