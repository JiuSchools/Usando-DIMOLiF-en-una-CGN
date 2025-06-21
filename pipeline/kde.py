import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from numpy.linalg import norm

def kde_1d_rbf(values, num_bins=8, bandwidth=0.2, value_range=(0, 1), circular=False):
    """
    Aplica KDE usando kernel RBF sobre un conjunto de valores (magnitudes o ángulos).

    Args:
        values (np.array): valores de entrada (1D)
        num_bins (int): número de bins del histograma
        bandwidth (float): sigma del kernel RBF
        value_range (tuple): rango esperado de los datos
        circular (bool): si es True, aplica wrapping para orientación angular

    Returns:
        np.array: histograma KDE normalizado (num_bins,)
    """
    if len(values) == 0:
        return np.zeros(num_bins)

    # Crear puntos de evaluación (centros de bin)
    centers = np.linspace(value_range[0], value_range[1], num_bins).reshape(-1, 1)
    values = values.reshape(-1, 1)

    if circular:
        # Expande valores circulares (para orientación)
        values = np.concatenate([values, values + 2*np.pi, values - 2*np.pi], axis=0)

    # Aplicar kernel RBF
    K = rbf_kernel(centers, values, gamma=1 / (2 * bandwidth**2))
    density = K.sum(axis=1)

    # Normalización L2
    return density / norm(density)
