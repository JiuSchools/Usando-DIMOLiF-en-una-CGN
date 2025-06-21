# Este modulo para divir los IF's en bloques (4x4) y seleccionar solamente los frames que contienen un umbral de STIP (N = 3)
from STCs_Computation import extract_stcs_from_frame
from load_interest_frames import load_if_data_from_csv
from STIP_points.load_video import load_video_to_array
import os


def resegment_video_by_T(video_volume, if_data, T, threshold=3):
    """
    Resecciona el video usando los IF y genera STCs para cada segmento.

    - Divide el rango [primer IF, último IF] en T partes. 
    - Agrupa los frames en cada parte.
    - Llama a extract_stcs_from_frame para cada frame.

    Retorna: lista de STCs (np.array de tamaño 10×10×5)
    """
    if not if_data:
        print("No hay Interest Frames para este video.")
        return []

    all_ifs = sorted(if_data.keys())
    start_if = all_ifs[0]
    end_if = all_ifs[-1]

    total_ifs = end_if - start_if + 1
    segment_size = total_ifs // T if T > 0 else 1

    stcs = []

    for t in range(T):
        seg_start = start_if + t * segment_size
        seg_end = start_if + (t + 1) * segment_size - 1 if t < T - 1 else end_if

        segment_ifs = [f for f in all_ifs if seg_start <= f <= seg_end]

        for frame_idx in segment_ifs:
            stip_points = if_data[frame_idx]
            stcs_block = extract_stcs_from_frame(
                frame_idx=frame_idx,
                stip_points=stip_points,
                video_volume=video_volume,
                threshold=threshold
            )
            stcs.extend(stcs_block)

    return stcs



def run_pipeline(csv_path, video_dir, T=8, threshold=3):
    """
    Ejecuta toda la tubería para todos los videos en el CSV.
    Retorna un diccionario con {video_name: [STCs]}
    """
    if_by_video = load_if_data_from_csv(csv_path)
    all_video_features = {}

    for video_name, if_data in if_by_video.items():
        video_path = os.path.join(video_dir, video_name)
        print(f"Procesando {video_name}...")
        try:
            video_volume = load_video_to_array(video_path)
            stcs = resegment_video_by_T(video_volume, if_data, T, threshold)
            all_video_features[video_name] = stcs
            print(f"{len(stcs)} STCs generados para {video_name}")
        except Exception as e:
            print(f"Error procesando {video_name}: {e}")

    return all_video_features