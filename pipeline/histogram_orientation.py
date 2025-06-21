import numpy as np
from kde import kde_1d_rbf
from magnitud_orientation import compute_mag_ori_from_stc

def generate_orientation_histogram_from_stc(stc, num_bins=8, bandwidth=0.2, threshold=0.05):
    """
    Calcula el histograma de orientación (8 bins) basado en coincidencia con la densidad KDE.

    Args:
        stc (np.array): bloque 10x10x5 del video
        num_bins (int): número de bins del histograma
        bandwidth (float): parámetro de suavizado del kernel RBF
        threshold (float): umbral mínimo de coincidencia de densidad para conservar valores

    Returns:
        np.array: histograma normalizado (L2) de tamaño (8,)
    """
    # 1. Calcular magnitudes y orientaciones
    mags, oris = compute_mag_ori_from_stc(stc)  # (10,10,4), (10,10,4)
    oris_flat = oris.flatten()

    # Convertir orientación de radianes [-π, π] a [0, 2π]
    oris_flat = np.mod(oris_flat, 2*np.pi)

    # 2. Aplicar KDE sobre todas las orientaciones
    kde_density = kde_1d_rbf(
        values=oris_flat,
        num_bins=num_bins,
        bandwidth=bandwidth,
        value_range=(0, 2*np.pi),
        circular=True
    )

    # 3. Seleccionar solo las orientaciones cuyo valor esté cerca del estimado por KDE
    # Estimamos la densidad esperada para cada valor real:
    centers = np.linspace(0, 2*np.pi, num_bins)
    bin_indices = np.digitize(oris_flat, centers, right=False) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    filtered_oris = oris_flat[kde_density[bin_indices] >= threshold]

    # 4. Generar histograma sobre los valores filtrados
    hist, _ = np.histogram(filtered_oris, bins=num_bins, range=(0, 2*np.pi))

    # 5. Normalizar el histograma (L2)
    norm_val = np.linalg.norm(hist)
    if norm_val > 0:
        hist = hist / norm_val

    return hist
