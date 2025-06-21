# Crea los cubos espacio temporales.

import numpy as np

def add_zero_vector(spatial_size=10, temporal_size=5):
    """
    Retorna un vector 0 del mismo tamaño que un STC válido.
    """
    return [np.zeros((spatial_size, spatial_size, temporal_size), dtype=np.float32)]



def extract_stcs_from_frame(frame_idx, stip_points, video_volume, threshold=3, grid_size=(4, 4), spatial_size=10, temporal_size=5):
    """
    Divide un Interest Frame en bloques 4x4, y genera STCs (10x10x5) para los bloques con suficiente densidad de STIP.

    Args:
        frame_idx (int): índice temporal del frame.
        stip_points (list of (x, y)): coordenadas de puntos STIP en ese frame.
        video_volume (np.array): video completo (alto, ancho, tiempo).
        threshold (int): mínimo de puntos por bloque.
        grid_size (tuple): cuadrícula (filas, columnas).
        spatial_size (int): tamaño espacial del STC.
        temporal_size (int): tamaño temporal del STC.

    Returns:
        list of np.array: STCs extraídos o un vector 0 si no hay bloques válidos.
    """
    H, W, T = video_volume.shape
    rows, cols = grid_size
    h_block = H // rows
    w_block = W // cols

    # Inicializar bloques vacíos
    blocks = [[[] for _ in range(cols)] for _ in range(rows)]

    # Asignar puntos STIP a bloques
    for x, y in stip_points:
        row = min(x // h_block, rows - 1)
        col = min(y // w_block, cols - 1)
        blocks[row][col].append((x, y))

    valid_stcs = []

    for i in range(rows):
        for j in range(cols):
            points = blocks[i][j]
            if len(points) >= threshold:
                # Calcular centroide
                xs, ys = zip(*points)
                cx, cy = int(np.mean(xs)), int(np.mean(ys))

                # Definir recorte espacial y temporal
                half_spatial = spatial_size // 2
                half_temporal = temporal_size // 2

                x_start = max(cx - half_spatial, 0)
                x_end = min(cx + half_spatial, H)
                y_start = max(cy - half_spatial, 0)
                y_end = min(cy + half_spatial, W)
                t_start = max(frame_idx - half_temporal, 0)
                t_end = min(frame_idx + half_temporal + 1, T)

                patch = video_volume[x_start:x_end, y_start:y_end, t_start:t_end]

                # Rellenar si es necesario
                stc = np.zeros((spatial_size, spatial_size, temporal_size), dtype=video_volume.dtype)
                stc[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch

                valid_stcs.append(stc)

    if not valid_stcs:
        return add_zero_vector(spatial_size, temporal_size)
    return valid_stcs
