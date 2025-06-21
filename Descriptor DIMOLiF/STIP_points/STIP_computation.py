import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, maximum_filter


# 1. Derivadas con suavizado Gaussiano
def compute_derivatives(video_block, sigma, tau):
    L = gaussian_filter(video_block, sigma=[sigma, sigma, 0])
    L = gaussian_filter1d(L, sigma=tau, axis=2)
    Lx = np.gradient(L, axis=0)
    Ly = np.gradient(L, axis=1)
    Lt = np.gradient(L, axis=2)
    return Lx, Ly, Lt

# 2. Tensor de segundo momento
def compute_structure_tensor(Lx, Ly, Lt, sigma_i, tau_i):
    def smooth(M): return gaussian_filter(M, sigma=[sigma_i, sigma_i, tau_i])
    return {
        'Lx2': smooth(Lx * Lx),
        'Ly2': smooth(Ly * Ly),
        'Lt2': smooth(Lt * Lt),
        'LxLy': smooth(Lx * Ly),
        'LxLt': smooth(Lx * Lt),
        'LyLt': smooth(Ly * Lt)
    }

# 3. Respuesta tipo Harris
def compute_harris_response(tensor, k=0.005):
    Lx2, Ly2, Lt2 = tensor['Lx2'], tensor['Ly2'], tensor['Lt2']
    LxLy, LxLt, LyLt = tensor['LxLy'], tensor['LxLt'], tensor['LyLt']
    det = (
        Lx2 * (Ly2 * Lt2 - LyLt**2) -
        LxLy * (LxLy * Lt2 - LxLt * LyLt) +
        LxLt * (LxLy * LyLt - LxLt * Ly2)
    )
    trace = Lx2 + Ly2 + Lt2
    return det - k * (trace ** 3)

# 4. Detección de picos locales
def detect_interest_points(H, threshold_rel=0.01):
    maxima = maximum_filter(H, size=(3, 3, 3))
    peaks = (H == maxima) & (H > threshold_rel * H.max())
    return np.argwhere(peaks)