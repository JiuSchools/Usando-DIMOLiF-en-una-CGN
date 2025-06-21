import os
from tqdm import tqdm
import numpy as np
import gc

from resegment_by_T import resegment_video_by_T
from load_interest_frames import load_if_data_from_csv
from STIP_points.load_video import load_video_to_array
from histogram_orientation import generate_orientation_histogram_from_stc


def extract_and_save_histograms_skip_existing(csv_path, video_dir, output_dir,
                                               T=8, threshold=3, num_bins=8,
                                               bandwidth=0.2, kde_threshold=0.05):
    """
    Igual que `extract_and_save_histograms`, pero omite los videos ya procesados.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("Cargando STIPs desde CSV...")
    if_by_video = load_if_data_from_csv(csv_path)

    for video_name, if_data in if_by_video.items():
        npy_name = video_name.replace('.mp4', '.npy')
        output_path = os.path.join(output_dir, npy_name)

        if os.path.exists(output_path):
            print(f"Ya existe {npy_name}, se omite.")
            continue

        print(f"\nProcesando {video_name}")
        video_path = os.path.join(video_dir, video_name)

        try:
            print(" - Cargando video...")
            try:
                video_volume = load_video_to_array(video_path)
            except Exception as e:
                print(f"Video omitido (fue eliminado): {e}")
                continue
            print(" - Extrayendo STCs...")
            stcs = resegment_video_by_T(video_volume, if_data, T, threshold)

            print(" - Generando histogramas...")
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
            np.save(output_path, histograms)
            print(f"Guardado en: {output_path}")
            del video_volume
            del stcs
            del histograms
            gc.collect()

        except Exception as e:
            print(f"Error procesando {video_name}: {e}")