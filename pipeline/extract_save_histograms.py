import os
import numpy as np
from tqdm import tqdm
from load_interest_frames import load_if_data_from_csv
from resegment_by_T import resegment_video_by_T
from STIP_points.load_video import load_video_to_array
from histogram_orientation import generate_orientation_histogram_from_stc

def extract_and_save_histograms(csv_path, video_dir, output_dir,
                                 T=8, threshold=3, num_bins=8,
                                 bandwidth=0.2, kde_threshold=0.05):
    """
    Extrae histogramas de orientación desde STCs y los guarda como archivos .npy por video.

    Args:
        csv_path (str): Ruta al CSV con STIPs
        video_dir (str): Carpeta donde están los videos
        output_dir (str): Carpeta donde guardar los .npy
        T (int): Número de segmentos por video
        threshold (int): Mínimo de STIPs por bloque
        num_bins (int): Número de bins del histograma
        bandwidth (float): ancho de banda del RBF kernel
        kde_threshold (float): umbral mínimo para retener valores reales
    """
    os.makedirs(output_dir, exist_ok=True)
    print("Cargando datos del CSV...")
    if_by_video = load_if_data_from_csv(csv_path)

    for video_name, if_data in if_by_video.items():
        print(f"\nProcesando video: {video_name}")
        video_path = os.path.join(video_dir, video_name)

        try:
            print(" - Cargando video como volumen...")
            video_volume = load_video_to_array(video_path)

            print(" - Extrayendo STCs...")
            stcs = resegment_video_by_T(video_volume, if_data, T, threshold)
            print(f"   → {len(stcs)} STCs extraídos.")

            print(" - Generando histogramas de orientación...")
            histograms = []
            for stc in tqdm(stcs, desc="   STCs procesados"):
                hist = generate_orientation_histogram_from_stc(
                    stc,
                    num_bins=num_bins,
                    bandwidth=bandwidth,
                    threshold=kde_threshold
                )
                histograms.append(hist)

            histograms = np.stack(histograms)
            output_path = os.path.join(output_dir, video_name.replace('.mp4', '.npy'))
            np.save(output_path, histograms)
            print(f"Guardado en: {output_path}")

        except Exception as e:
            print(f"Error procesando {video_name}: {e}")
